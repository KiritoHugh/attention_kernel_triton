import torch
import torch_npu
DEVICE_STR = "npu"
import triton
import triton.language as tl
import os

import math
# 16 = 16



@triton.jit
def sparsetoken_flash_attention_decode_paged_kernel(
    # data ptr
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    # 
    O_ptr,
    softmax_scale,

    # strides
    stride_paged_kv_cache_pages,
    stride_paged_kv_cache_2,
    stride_paged_kv_cache_H,
    stride_paged_kv_cache_P,
    stride_paged_kv_cache_D,

    stride_q_B,
    stride_q_H_GQA,
    stride_q_1,
    stride_q_D,

    stride_sparse_ind_B,
    stride_sparse_ind_H,

    stride_sparse_nnz_B,
    stride_sparse_nnz_H,

    stride_O_B,
    stride_O_H_GQA,
    stride_O_1,
    stride_O_D,


    # shapes
    BATCH_SIZE,
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_groups_size: tl.constexpr,
    PAGE_SIZE: tl.constexpr,

    # others
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
):
    '''
    q: [B, qo_heads, 16, head_dim]
    paged_kv_cache: [all_pages, 2, kv_heads, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    sparse_ind: [B, H, L_max]
    sparse_nnz: [B, H, 1]
    O: [B, qo_heads, 16, head_dim]
    '''

    B_H_id = tl.program_id(0)
    out_batch_id = B_H_id // NUM_QO_HEADS
    out_head_id = B_H_id % NUM_QO_HEADS
    kv_head_id = out_head_id // GQA_groups_size

    qo_offset = out_batch_id * stride_q_B + out_head_id * stride_q_H_GQA

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_O_1, stride_O_D),
        # 
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_q_1, stride_q_D),
        # 
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    q_block = tl.load(q_block_ptr)
    tmp_O_block = tl.zeros((16, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((16,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((16,), dtype=tl.float32)

    b_h_nzz = tl.load(sparse_nnz_ptr + out_batch_id * stride_sparse_nnz_B + out_head_id * stride_sparse_nnz_H)
    b_h_ind_ptr_base = sparse_ind_ptr + out_batch_id * stride_sparse_ind_B + out_head_id * stride_sparse_ind_H

    # shared_K = tl.shared_memory((HEAD_DIM, KV_SEQ_BLOCK_SIZE), dtype=paged_kv_cache_ptr.dtype.element_ty)
    # shared_V = tl.shared_memory((KV_SEQ_BLOCK_SIZE, HEAD_DIM), dtype=paged_kv_cache_ptr.dtype.element_ty)

    page_idx_start = tl.load(kv_page_indptr_ptr + out_batch_id)

    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    for ind_start_idx in range(0, b_h_nzz, KV_SEQ_BLOCK_SIZE):
        mask = ind_start_idx + KV_ranges < b_h_nzz
        # load k,v to shared memory
        token_idx = tl.load(b_h_ind_ptr_base + ind_start_idx + KV_ranges, mask=mask, other=0)
        page_idx = token_idx // PAGE_SIZE
        offset_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(kv_page_indices_ptr + page_idx_start + page_idx, mask=mask, other=0)

        k_ptr = paged_kv_cache_ptr + page_id * stride_paged_kv_cache_pages + \
                kv_head_id * stride_paged_kv_cache_H + \
                offset_in_page * stride_paged_kv_cache_P
        
        v_ptr = paged_kv_cache_ptr + page_id * stride_paged_kv_cache_pages + \
                1 * stride_paged_kv_cache_2 + \
                kv_head_id * stride_paged_kv_cache_H + \
                offset_in_page * stride_paged_kv_cache_P

        shared_K = tl.load(
            k_ptr[None,:] + tl.arange(0, HEAD_DIM)[:, None] * stride_paged_kv_cache_D,
            mask=mask[None, :],
            other=0.0
        )

        shared_V = tl.load(
            v_ptr[:,None] + tl.arange(0, HEAD_DIM)[None, :] * stride_paged_kv_cache_D,
            mask=mask[:, None],
            other=0.0
        )

        # tl.barrier()

        # compute attention
        QK_block = tl.dot(q_block, shared_K)
        mask = ind_start_idx + KV_ranges < b_h_nzz
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
        P_block = P_block.to(tl.float16)
        tmp_O_block = tmp_O_block * alpha[:, None] 
        tmp_O_block = tl.dot(P_block, shared_V, tmp_O_block)

        # 
        tmp_m_i = m_ij

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))
    

def sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):
    '''
    q: [B, qo_heads, 1, head_dim]
    '''
    q = q.repeat_interleave(16, dim=2)  # [B, qo_heads, 16, head_dim]

    B, num_qo_heads, _, head_dim = q.shape
    num_kv_heads = paged_kv_cache.shape[2]
    groups = num_qo_heads // num_kv_heads
    page_size = paged_kv_cache.shape[3]

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # allocate output
    O = torch.zeros((B, num_qo_heads, 16, head_dim), device=q.device, dtype=q.dtype)

    grid = (
        B * num_qo_heads,  # one block per (batch, head)
        1,
        1
    )
    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 16))


    sparsetoken_flash_attention_decode_paged_kernel[grid](
        # data ptr
        q,
        paged_kv_cache,
        kv_page_indptr,
        kv_page_indices,
        sparse_ind,
        sparse_nnz,
        #
        O,
        softmax_scale,
        # strides
        paged_kv_cache.stride(0),
        paged_kv_cache.stride(1),
        paged_kv_cache.stride(2),
        paged_kv_cache.stride(3),
        paged_kv_cache.stride(4),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        sparse_ind.stride(0),
        sparse_ind.stride(1),
        sparse_nnz.stride(0),
        sparse_nnz.stride(1),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        # shapes
        B,
        num_qo_heads,
        head_dim,
        groups,
        page_size,
        # others
        VAR_KV_SEQ_BLK_SIZE,
    )
    return O[:, :, 0, :].contiguous()  # [B, qo_heads, head_dim]

# HERE IS THE BASELINE
def sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):

    '''
    q: [B, qo_heads, 1, head_dim]

    paged_kv_cache: [B, 2, H, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    kv_last_page_len: [B]  (len <= page_size)

    sparse_ind: [B, H, L_max]  
    sparse_nnz: [B, H, 1] 
    '''
    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[3]
    num_kv_heads = paged_kv_cache.shape[2]
    page_size = paged_kv_cache.shape[3]
    groups = num_qo_heads // num_kv_heads

    k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]  # [pages, heads, page_size, head_dim]
    
    outputs = []
    for b in range(batch_size):
        batch_outputs = []
        for h in range(num_qo_heads):
            nnz = sparse_nnz[b, h].item()
            if nnz == 0:
                # Empty sequence
                batch_outputs.append(torch.zeros((1, head_dim), device=q.device, dtype=q.dtype))
                continue
            
            seq_head_k_parts = []
            seq_head_v_parts = []
            indices = sparse_ind[b, h, :nnz]
            kv_h = h // groups  # handle GQA
            for i in range(nnz):
                token_idx = indices[i].item()
                page_idx = token_idx // page_size
                page_id = kv_page_indices[kv_page_indptr[b].item() + page_idx]
                offset_in_page = token_idx % page_size
                seq_head_k_parts.append(k_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])  # [1, head_dim]
                seq_head_v_parts.append(v_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])
            seq_head_k = torch.cat(seq_head_k_parts, dim=0).unsqueeze(0)  # [1, seq_len, head_dim]
            seq_head_v = torch.cat(seq_head_v_parts, dim=0).unsqueeze(0)

            # Attention computation
            q_bh = q[b, h:h+1]  # [1, 1, head_dim]
            scores = torch.matmul(q_bh, seq_head_k.permute(0, 2, 1)) / math.sqrt(head_dim)  # [1, 1, seq_len]
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, seq_head_v)  # [1, 1, head_dim]
            batch_outputs.append(out.squeeze(1))  # [1, head_dim]
        outputs.append(torch.cat(batch_outputs, dim=0).unsqueeze(0))  # [1, qo_heads, head_dim]

    return torch.cat(outputs, dim=0) # [B, qo_heads, head_dim]


def test_op_decode_paged_sparsetoken(GQA_group_size = 4, dtype=torch.float16):
    pass

    device = DEVICE_STR
    # Test parameters
    num_kv_heads = 8
    num_qo_heads = num_kv_heads * GQA_group_size
    head_dim = 256
    page_size = 256
    max_num_pages = 1024
    max_num_pages_per_seq = 512
    
    # Create paged KV cache
    paged_kv_cache = torch.randn(max_num_pages, 2, num_kv_heads, page_size, head_dim, 
                                 device=device, dtype=dtype)
    
    '''
    sparse_ind: [B, H, L_max]  
    sparse_nnz: [B, H, 1] 
    '''

    # # Create metadata
    # batch_size = 3
    # kv_page_indptr = torch.tensor([0, 4, 9, 12], dtype=torch.int32, device=device)  # 
    # kv_page_indices = torch.tensor([0, 1, 3, 5, 
    #                                 2, 8, 9, 10, 11, 
    #                                 6, 7 ,4], dtype=torch.int32, device=device)
    # kv_last_page_len = torch.tensor([63, 22, 55], dtype=torch.int32, device=device)  # 
    
    # Create metadata
    batch_size = 3
    num_pages_per_seq = torch.tensor([270, 350, 321], dtype=torch.int32, device=device)
    kv_page_indices = torch.tensor(range(sum(num_pages_per_seq)), dtype=torch.int32, device=device)
    # permute to simulate random pages
    kv_page_indices = kv_page_indices[torch.randperm(kv_page_indices.shape[0], device=device)]
    kv_page_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
    kv_page_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    kv_last_page_len = torch.tensor([63, 22, 55], dtype=torch.int32, device=device)  # 
    if torch.any(kv_last_page_len > page_size):
        kv_last_page_len.fill_(page_size-1)

    # random sparse, sparse ratio 0.5
    kept_ratio = 0.02
    L_max = max_num_pages_per_seq * page_size
    original_nzz = torch.zeros((batch_size, num_qo_heads, 1), dtype=torch.int32, device=device)
    sparse_ind = torch.zeros((batch_size, num_qo_heads, L_max), dtype=torch.int32, device=device) - 1
    # 
    for b in range(batch_size):
        original_nzz[b] = (kv_page_indptr[b+1] - kv_page_indptr[b]) * page_size - (page_size - kv_last_page_len[b])
    # print("original_nzz:", original_nzz)
    for bh in range(batch_size*num_qo_heads):
        b = bh // num_qo_heads
        h = bh % num_qo_heads
        # randomly drop by probability sparse_ratio
        sample_prob = torch.rand((original_nzz[b][h].item(),), device=device)
        kept_mask = sample_prob < kept_ratio
        kept_nnz = kept_mask.sum().item()
        if kept_nnz == 0:
            kept_nnz = 1
            kept_mask[0] = True
        original_nzz[b][h] = kept_nnz
        sparse_ind[b, h, :kept_nnz] = torch.nonzero(kept_mask, as_tuple=False).squeeze(-1)  # [H, L_max]
    sparse_nnz = original_nzz
    # 
    # print("sparse_ind:", sparse_ind)
    # print("sparse_nnz:", sparse_nnz)
    print("real kept ratio:", sparse_nnz.float().mean().item() / ((kv_page_indptr[1:] - kv_page_indptr[:-1]).float().mean().item() * page_size))



    # Create query
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)


    def triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):
        return sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)

    # compare 
    ref_O = sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)
    print("shape of ref_O:", ref_O.shape)

    # just for evaluating the baseline in pytorch npu
    print("complete baseline running.")
    
    # tri_O = triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)
    # print("shape of tri_O:", tri_O.shape)
    # # triton_O how many nan? its ratio?
    # print("Number of NaNs in triton_O:", torch.isnan(tri_O).sum().item())
    # print("Ratio of NaNs in triton_O:", torch.isnan(tri_O).float().mean().item())


    # rtol = 0.0
    # atol = 1e-2
    # # print the max absolute values
    # print("Max absolute values - ref:", torch.max(torch.abs(ref_O)).item(), " tri:", torch.max(torch.abs(tri_O)).item())

    # # print the max absolute difference
    # print("Max absolute difference:", torch.max(torch.abs(ref_O - tri_O)).item())
    # assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"


    # benchmark
    print("Benchmarking reference implementation...")
    ref_ms = triton.testing.do_bench(lambda: sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    # print("Benchmarking Triton implementation...")
    # tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz))
    # print(f"Triton implementation: {tri_ms:.3f} ms")

    # print(f"Speedup: {ref_ms / tri_ms:.3f}x")

if __name__ == "__main__":
    test_op_decode_paged_sparsetoken(GQA_group_size=4, dtype=torch.float16)


'''
Test on NVIDIA RTX 5000 Ada Generation

Output:
```
real kept ratio: 0.01997607291090003
shape of ref_O: torch.Size([3, 32, 256])
shape of tri_O: torch.Size([3, 32, 256])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
Max absolute values - ref: 0.183837890625  tri: 0.183837890625
Max absolute difference: 0.0001220703125
Benchmarking reference implementation...
Reference implementation: 7751.282 ms
Benchmarking Triton implementation...
Triton implementation: 0.846 ms
Speedup: 9162.089x
```
'''
