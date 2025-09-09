
import torch
import triton
import triton.language as tl
import os

import math
redundant_len = 16

@triton.jit
def inner_kernel(
    # query block
    Q_block,
    # 
    # K V ptrs
    K_block_ptr,
    V_block_ptr,
    # 
    # block size
    KV_SEQ_BLOCK_SIZE,
    # 
    # locals
    tmp_O_block,
    tmp_m_i,
    tmp_l_i,
    # 
    SEQ_LEN,
    # 
    # other
    softmax_scale,
    KV_ranges,
):
    seq_lo, seq_hi = 0, SEQ_LEN
    # loop K V by KV_SEQ_BLOCK_SIZE
    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        # compute q@k
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        mask =  SEQ_LEN > KV_ranges + start_kv
        QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)[None, :]
        # mantain the max value 
        m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1))
        QK_block -= m_ij[:, None]

        # compute exp, sumofexp, 
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)

        # record alpha for sumofexp correction, and correct sumofexp
        alpha = tl.math.exp(tmp_m_i - m_ij)
        tmp_l_i = tmp_l_i * alpha + l_ij

        # compute output
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        tmp_O_block = tmp_O_block * alpha[:, None] 
        tmp_O_block = tl.dot(P_block, V_block, tmp_O_block)

        # 
        tmp_m_i = m_ij

        # advance the loop
        V_block_ptr = tl.advance(V_block_ptr, (KV_SEQ_BLOCK_SIZE, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, KV_SEQ_BLOCK_SIZE))
    return tmp_O_block, tmp_m_i, tmp_l_i

@triton.jit
def flash_attention_paged_kernel(
    # data ptr
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    kv_last_page_len_ptr,
    # 
    O_ptr,
    softmax_scale,

    # stride
    stride_q_B,
    stride_q_H_GQA,
    stride_q_1,
    stride_q_D,

    stride_paged_kv_cache_B,
    stride_paged_kv_cache_2,
    stride_paged_kv_cache_H,
    stride_paged_kv_cache_page,
    stride_paged_kv_cache_D,

    stride_O_B,
    stride_O_H_GQA,
    stride_O_1,
    stride_O_D,

    # shapes
    BATCH_SIZE,
    H_GQA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_group_size: tl.constexpr,
    page_size: tl.constexpr,

    # block
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
):
    '''
    q: (B, H*GQA_group_size, redundant_len, D)

    paged_kv_cache: (all_num_pages, 2, H, page_size, D) (2 for k and v)
    kv_page_indptr: (B+1) (int32)
    kv_page_indices: (total_num_pages) (int32)
    kv_last_page_len: (B) (int32)

    O: (B, H*GQA_group_size, redundant_len, D)


    '''



    B_H_GQA_id = tl.program_id(0)
    out_batch_id = B_H_GQA_id // H_GQA
    out_head_id = B_H_GQA_id % H_GQA 
    kv_head_id = out_head_id // GQA_group_size  # head id for kv

    indptr_start = tl.load(kv_page_indptr_ptr + out_batch_id)
    indptr_end = tl.load(kv_page_indptr_ptr + out_batch_id + 1)
    num_pages = indptr_end - indptr_start

    qo_offset = out_batch_id * stride_q_B + out_head_id * stride_q_H_GQA
    qo_offset_O = out_batch_id * stride_O_B + out_head_id * stride_O_H_GQA


    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset_O,
        shape = (redundant_len, HEAD_DIM),
        strides = (stride_O_1, stride_O_D),
        # 
        block_shape = (redundant_len, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (redundant_len, HEAD_DIM),
        strides = (stride_q_1, stride_q_D),
        # 
        block_shape = (redundant_len, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )


    q_block = tl.load(q_block_ptr)
    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    # 
    tmp_O_block = tl.zeros((redundant_len, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    # not yet handle num_pages == 0 case
    for i in range(num_pages):
        page_idx = tl.load(kv_page_indices_ptr + indptr_start + i)
        last_page_len = 0
        if i == num_pages - 1:
            last_page_len = tl.load(kv_last_page_len_ptr + out_batch_id)
        else:
            last_page_len = page_size

        # move offset for k, v from paged_kv_cache
        k_ptr_offset = page_idx * stride_paged_kv_cache_B + kv_head_id * stride_paged_kv_cache_H
        v_ptr_offset = page_idx * stride_paged_kv_cache_B + stride_paged_kv_cache_2 + kv_head_id * stride_paged_kv_cache_H

        # needed shape (page_size, D), frankly is just lie (s,D)
        K_block_ptr = tl.make_block_ptr(
            base = paged_kv_cache_ptr + k_ptr_offset,
            shape = (HEAD_DIM, page_size),
            strides = (
                stride_paged_kv_cache_D,
                stride_paged_kv_cache_page,
                ),
            # 
            block_shape = (HEAD_DIM, KV_SEQ_BLOCK_SIZE),
            offsets = (0, 0),
            # 
            order = (0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base = paged_kv_cache_ptr + v_ptr_offset,
            shape = (page_size, HEAD_DIM),
            strides = (
                stride_paged_kv_cache_page, 
                stride_paged_kv_cache_D),
            # 
            block_shape = (KV_SEQ_BLOCK_SIZE, HEAD_DIM),
            offsets = (0, 0),
            # 
            order = (1, 0),
        )



        # call programs 
        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            q_block,
            K_block_ptr,
            V_block_ptr,
            KV_SEQ_BLOCK_SIZE,
            tmp_O_block,
            tmp_m_i,
            tmp_l_i,
            last_page_len,
            softmax_scale,
            KV_ranges,
        )

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))



def flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len):
    """
    q: [B, H*GQA_group_size, 1, D]
    paged_kv_cache: [all_num_pages, 2, H, page_size, D] (2 for k and v)
    kv_page_indptr: [B+1] (int32)
    kv_page_indices: [total_num_pages] (int32)
    kv_last_page_len: [B] (int32)
    """
    q = q.repeat_interleave(redundant_len, dim=2)  # [B, H*GQA_group_size, redundant_len, D]

    B, H_GQA, _, D = q.shape
    H = paged_kv_cache.shape[2]
    GQA_group_size = H_GQA // H
    page_size = paged_kv_cache.shape[3]

    # softmax_scale = 1.0 / (D**0.5)
    softmax_scale = float(1.0 / math.sqrt(D))

    # allocate output
    O = torch.empty((B, H_GQA, redundant_len, D), dtype=q.dtype, device=q.device)

    grid = (
        B * H_GQA,
        1,
        1,
    )

    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 16))

    assert VAR_KV_SEQ_BLK_SIZE <= page_size and page_size % VAR_KV_SEQ_BLK_SIZE == 0, "page_size must be multiple of VAR_KV_SEQ_BLK_SIZE"

    flash_attention_paged_kernel[grid](
        q,
        paged_kv_cache,
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        O,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        paged_kv_cache.stride(0), paged_kv_cache.stride(1), paged_kv_cache.stride(2), paged_kv_cache.stride(3), paged_kv_cache.stride(4),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H_GQA, D, GQA_group_size, page_size,
        VAR_KV_SEQ_BLK_SIZE,
    )

    return O[ :, :, 0, : ].contiguous()  # [B, H*GQA_group_size, 1, D]



def naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len):
    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[3]
    num_kv_heads = paged_kv_cache.shape[2]

    k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]  # [pages, heads, page_size, head_dim]
    
    outputs = []
    for b in range(batch_size):
        page_start = kv_page_indptr[b].item()
        page_end = kv_page_indptr[b + 1].item()
        
        if page_start == page_end:
            # Empty sequence
            outputs.append(torch.zeros_like(q[b:b+1]))
            continue
            
        # Reconstruct k, v for this sequence
        seq_pages = kv_page_indices[page_start:page_end]
        seq_k_parts = []
        seq_v_parts = []
        
        for i, page_idx in enumerate(seq_pages):
            k_page = k_cache[page_idx]  # [heads, page_size, head_dim]
            v_page = v_cache[page_idx]
            
            if i == len(seq_pages) - 1:  # Last page
                last_len = kv_last_page_len[b].item()
                k_page = k_page[:, :last_len, :]
                v_page = v_page[:, :last_len, :]
                
            seq_k_parts.append(k_page)
            seq_v_parts.append(v_page)
        
        seq_k = torch.cat(seq_k_parts, dim=1)  # [kv_heads, seq_len, head_dim]
        seq_v = torch.cat(seq_v_parts, dim=1)
        
        # Handle GQA
        if num_qo_heads != num_kv_heads:
            groups = num_qo_heads // num_kv_heads
            seq_k = seq_k.repeat_interleave(groups, dim=0)
            seq_v = seq_v.repeat_interleave(groups, dim=0)
        
        # Attention computation
        q_b = q[b]  # [qo_heads, 1, head_dim]
        scores = torch.matmul(q_b, seq_k.permute(0, 2, 1)) / math.sqrt(head_dim)  # [qo_heads, 1, seq_len]
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, seq_v.permute(0, 1, 2)).squeeze(1)  # [qo_heads, head_dim]
        outputs.append(out.unsqueeze(0))
        
    return torch.cat(outputs, dim=0)


def test_op_decode_paged(GQA_group_size = 2, dtype=torch.float16):
    pass

    device = "cuda"
    # Test parameters
    num_kv_heads = 2
    num_qo_heads = num_kv_heads * GQA_group_size
    head_dim = 64
    page_size = 64
    max_num_pages = 16
    
    # Create paged KV cache
    paged_kv_cache = torch.randn(max_num_pages, 2, num_kv_heads, page_size, head_dim, 
                                 device=device, dtype=dtype)
    
    # Create metadata
    batch_size = 3
    kv_page_indptr = torch.tensor([0, 4, 5, 8], dtype=torch.int32, device=device)  # 
    kv_page_indices = torch.tensor([0, 1, 3, 5, 
                                    2, 
                                    6, 7 ,4], dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([2, 22, 3], dtype=torch.int32, device=device)  # 

    # batch_size = 4
    # kv_page_indptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)  # 
    # kv_page_indices = torch.tensor([0, 
    #                                 1, 
    #                                 2,
    #                                 3], dtype=torch.int32, device=device)
    # kv_last_page_len = torch.tensor([64, 64, 64, 64], dtype=torch.int32, device=device)  # 


    # Create query
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)


    def triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len):
        return flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len)


    # compare 
    ref_O = naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len)
    print("shape of ref_O:", ref_O.shape)
    tri_O = triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len)
    print("shape of triton_O:", tri_O.shape)
    # triton_O how many nan? its ratio?
    print("Number of NaNs in triton_O:", torch.isnan(tri_O).sum().item())
    print("Ratio of NaNs in triton_O:", torch.isnan(tri_O).float().mean().item())

    rtol = 0.0
    atol = 1e-2
    # print the max absolute values
    print("Max absolute values - ref:", torch.max(torch.abs(ref_O)).item(), " tri:", torch.max(torch.abs(tri_O)).item())

    # print the max absolute difference
    print("Max absolute difference:", torch.max(torch.abs(ref_O - tri_O)).item())
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"

    # benchmark
    print("Benchmarking reference implementation...")
    ref_ms = triton.testing.do_bench(lambda: naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    print("Benchmarking Triton implementation...")
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len))
    print(f"Triton implementation: {tri_ms:.3f} ms")

    print(f"Speedup: {ref_ms / tri_ms:.3f}x")



if __name__ == "__main__":
    test_op_decode_paged()


'''
Test on NVIDIA RTX 5000 Ada Generation

Output:
```
shape of ref_O: torch.Size([3, 4, 64])
shape of triton_O: torch.Size([3, 4, 64])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
Max absolute values - ref: 1.0302734375  tri: 1.0302734375
Max absolute difference: 0.0009765625
Benchmarking reference implementation...
Reference implementation: 0.763 ms
Benchmarking Triton implementation...
Triton implementation: 0.017 ms
Speedup: 44.067x
```
'''
