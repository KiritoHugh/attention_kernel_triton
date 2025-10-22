import torch
import triton
import triton.language as tl
import os

import math
redundant_len = 16



@triton.jit
def magicpig_sparsetoken_flash_attention_decode_paged_kernel(
    # data ptr
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    K_ptr,  # [B, H] - K parameter per head
    L_ptr,  # [B, H] - L parameter per head
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

    stride_K_B,
    stride_K_H,

    stride_L_B,
    stride_L_H,

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
    q: [B, qo_heads, redundant_len, head_dim]
    paged_kv_cache: [all_pages, 2, kv_heads, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    sparse_ind: [B, H, L_max]
    sparse_nnz: [B, H, 1]
    K_ptr: [B, H] - K parameter per head
    L_ptr: [B, H] - L parameter per head
    O: [B, qo_heads, redundant_len, head_dim]
    '''

    B_H_id = tl.program_id(0)
    out_batch_id = B_H_id // NUM_QO_HEADS
    out_head_id = B_H_id % NUM_QO_HEADS
    kv_head_id = out_head_id // GQA_groups_size

    qo_offset = out_batch_id * stride_q_B + out_head_id * stride_q_H_GQA

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
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
    
    # Load K and L parameters for this head
    K_val = tl.load(K_ptr + out_batch_id * stride_K_B + out_head_id * stride_K_H)
    L_val = tl.load(L_ptr + out_batch_id * stride_L_B + out_head_id * stride_L_H)
    
    # Compute q norm for normalization (convert to float32 for sqrt)
    q_block_f32 = q_block.to(tl.float32)
    q_norm = tl.sqrt(tl.sum(q_block_f32 * q_block_f32, axis=1))  # [redundant_len]
    
    tmp_O_block = tl.zeros((redundant_len, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    b_h_nzz = tl.load(sparse_nnz_ptr + out_batch_id * stride_sparse_nnz_B + out_head_id * stride_sparse_nnz_H)
    b_h_ind_ptr_base = sparse_ind_ptr + out_batch_id * stride_sparse_ind_B + out_head_id * stride_sparse_ind_H

    page_idx_start = tl.load(kv_page_indptr_ptr + out_batch_id)

    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    PI = 3.14159265358979323846
    
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

        # compute attention scores
        QK_block = tl.dot(q_block, shared_K)  # [redundant_len, KV_SEQ_BLOCK_SIZE]
        
        # Compute k norms for each token in the block (convert to float32 for sqrt)
        shared_K_f32 = shared_K.to(tl.float32)
        k_norm = tl.sqrt(tl.sum(shared_K_f32 * shared_K_f32, axis=0))  # [KV_SEQ_BLOCK_SIZE]
        
        # Compute normalized dot product: qk^T / (|q| * |k|)
        # Shape broadcasting: q_norm[:, None] * k_norm[None, :] -> [redundant_len, KV_SEQ_BLOCK_SIZE]
        qk_normalized = QK_block / (q_norm[:, None] * k_norm[None, :] + 1e-8)
        
        # Clamp to [-1, 1] to ensure valid arccos input (more conservative)
        qk_normalized = tl.where(qk_normalized > 1.0 - 1e-7, 1.0 - 1e-7, qk_normalized)
        qk_normalized = tl.where(qk_normalized < -1.0 + 1e-7, -1.0 + 1e-7, qk_normalized)
        
        # Compute p_i = 1 - (1/π) * arccos(qk_normalized)
        # Note: tl.acos gives arccos
        p_i = 1.0 - (1.0 / PI) * tl.math.acos(qk_normalized)
        
        # Clamp p_i to valid range [0, 1]
        p_i = tl.where(p_i > 1.0, 1.0, p_i)
        p_i = tl.where(p_i < 0.0, 0.0, p_i)
        
        # Compute u_i = 1 - (1 - p_i^K)^L - L*p_i^K * (1 - p_i^K)^(L-1)
        p_i_K = tl.math.pow(p_i, K_val)
        one_minus_p_i_K = 1.0 - p_i_K
        # Clamp to avoid numerical issues
        one_minus_p_i_K = tl.where(one_minus_p_i_K > 1.0, 1.0, one_minus_p_i_K)
        one_minus_p_i_K = tl.where(one_minus_p_i_K < 0.0, 0.0, one_minus_p_i_K)
        
        one_minus_p_i_K_L = tl.math.pow(one_minus_p_i_K, L_val)
        one_minus_p_i_K_L_minus_1 = tl.math.pow(one_minus_p_i_K, L_val - 1.0)
        
        u_i = 1.0 - one_minus_p_i_K_L - L_val * p_i_K * one_minus_p_i_K_L_minus_1
        
        # Clamp u_i to avoid log(0) or negative values
        u_i = tl.where(u_i <= 0.0, 1e-10, u_i)
        
        # Compute log(u_i)
        log_u_i = tl.math.log(u_i)
        
        # Apply attention formula: exp(qk^T/sqrt(d) - log(u_i))
        QK_block = QK_block * softmax_scale - log_u_i
        
        mask = ind_start_idx + KV_ranges < b_h_nzz
        QK_block = QK_block + tl.where(mask, 0.0, -1.0e6)[None, :]
        
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
    

def magicpig_sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):
    '''
    q: [B, qo_heads, 1, head_dim]
    K: [B, H] - K parameter per head
    L: [B, H] - L parameter per head
    '''
    q = q.repeat_interleave(redundant_len, dim=2)  # [B, qo_heads, redundant_len, head_dim]

    B, num_qo_heads, _, head_dim = q.shape
    num_kv_heads = paged_kv_cache.shape[2]
    groups = num_qo_heads // num_kv_heads
    page_size = paged_kv_cache.shape[3]

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # allocate output
    O = torch.zeros((B, num_qo_heads, redundant_len, head_dim), device=q.device, dtype=q.dtype)

    grid = (
        B * num_qo_heads,  # one block per (batch, head)
        1,
        1
    )
    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 16))


    magicpig_sparsetoken_flash_attention_decode_paged_kernel[grid](
        # data ptr
        q,
        paged_kv_cache,
        kv_page_indptr,
        kv_page_indices,
        sparse_ind,
        sparse_nnz,
        K,
        L,
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
        K.stride(0),
        K.stride(1),
        L.stride(0),
        L.stride(1),
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

def magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):

    '''
    q: [B, qo_heads, 1, head_dim]

    paged_kv_cache: [B, 2, H, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    kv_last_page_len: [B]  (len <= page_size)

    sparse_ind: [B, H, L_max]  
    sparse_nnz: [B, H, 1]
    K: [B, H] - K parameter per head
    L: [B, H] - L parameter per head
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

            # Get K and L for this head
            K_val = K[b, h].item()
            L_val = L[b, h].item()

            # Attention computation with MagicPIG normalization
            q_bh = q[b, h:h+1].float()  # [1, 1, head_dim] - convert to float32
            seq_head_k_f32 = seq_head_k.float()  # Convert to float32 for numerical stability
            
            # Compute QK scores
            scores = torch.matmul(q_bh, seq_head_k_f32.permute(0, 2, 1))  # [1, 1, seq_len]
            
            # Compute norms
            q_norm = torch.norm(q_bh, dim=-1, keepdim=True)  # [1, 1, 1]
            k_norm = torch.norm(seq_head_k_f32, dim=-1, keepdim=True)  # [1, seq_len, 1]
            
            # Compute normalized dot product
            qk_normalized = scores / (q_norm * k_norm.permute(0, 2, 1) + 1e-8)  # [1, 1, seq_len]
            qk_normalized = torch.clamp(qk_normalized, -1.0 + 1e-7, 1.0 - 1e-7)  # More conservative clamping
            
            # Compute p_i = 1 - (1/π) * arccos(qk_normalized)
            p_i = 1.0 - (1.0 / math.pi) * torch.acos(qk_normalized)
            
            # Clamp p_i to valid range [0, 1]
            p_i = torch.clamp(p_i, 0.0, 1.0)
            
            # Compute u_i = 1 - (1 - p_i^K)^L - L*p_i^K * (1 - p_i^K)^(L-1)
            p_i_K = torch.pow(p_i, K_val)
            one_minus_p_i_K = 1.0 - p_i_K
            # Clamp to avoid numerical issues
            one_minus_p_i_K = torch.clamp(one_minus_p_i_K, 0.0, 1.0)
            
            one_minus_p_i_K_L = torch.pow(one_minus_p_i_K, L_val)
            one_minus_p_i_K_L_minus_1 = torch.pow(one_minus_p_i_K, L_val - 1.0)
            
            u_i = 1.0 - one_minus_p_i_K_L - L_val * p_i_K * one_minus_p_i_K_L_minus_1
            u_i = torch.clamp(u_i, min=1e-10)
            
            # Compute log(u_i)
            log_u_i = torch.log(u_i)
            
            # Apply attention formula: exp(qk^T/sqrt(d) - log(u_i))
            scores = scores / math.sqrt(head_dim) - log_u_i
            
            weights = torch.softmax(scores, dim=-1)
            seq_head_v_f32 = seq_head_v.float()  # Convert to float32 for computation
            out = torch.matmul(weights, seq_head_v_f32)  # [1, 1, head_dim]
            batch_outputs.append(out.squeeze(1).to(q.dtype))  # [1, head_dim] - convert back to original dtype
        outputs.append(torch.cat(batch_outputs, dim=0).unsqueeze(0))  # [1, qo_heads, head_dim]

    return torch.cat(outputs, dim=0) # [B, qo_heads, head_dim]


def test_op_decode_paged_sparsetoken_magicpig(GQA_group_size = 4, dtype=torch.float16):
    pass

    device = "cuda"
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
    print("real kept ratio:", sparse_nnz.float().mean().item() / ((kv_page_indptr[1:] - kv_page_indptr[:-1]).float().mean().item() * page_size))

    # Create K and L parameters (different for each head)
    # K and L should be positive values, here we use random values for testing
    K = torch.rand(batch_size, num_qo_heads, device=device, dtype=torch.float32) * 5 + 1  # K in [1, 6]
    L = torch.rand(batch_size, num_qo_heads, device=device, dtype=torch.float32) * 5 + 1  # L in [1, 6]

    # Create query
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)


    def triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):
        return magicpig_sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)

    # compare 
    ref_O = magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)
    print("shape of ref_O:", ref_O.shape)
    tri_O = triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)
    print("shape of tri_O:", tri_O.shape)
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
    ref_ms = triton.testing.do_bench(lambda: magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    print("Benchmarking Triton implementation...")
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L))
    print(f"Triton implementation: {tri_ms:.3f} ms")

    print(f"Speedup: {ref_ms / tri_ms:.3f}x")

if __name__ == "__main__":
    test_op_decode_paged_sparsetoken_magicpig(GQA_group_size=4, dtype=torch.float16)


'''
MagicPIG Attention with Normalization

This implementation adds normalization to the sparse token decode attention based on the MagicPIG paper.

Key additions:
1. K and L parameters per head [B, H]
2. Computation of u_i normalization factor based on:
   - p_i = 1 - (1/π) * arccos(qk^T / (|q| · |k|))
   - u_i = 1 - (1 - p_i^K)^L - L*p_i^K * (1 - p_i^K)^(L-1)
3. Modified attention score: exp(qk^T/sqrt(d) - log(u_i)) / sum(...)

The normalization helps calibrate attention weights based on the similarity between query and key vectors.
'''


'''
$ python magicpig_sparsetoken_decode_flash_attention_redundant_var_len_paged_with_norm.py 
real kept ratio: 0.019930021389869657
shape of ref_O: torch.Size([3, 32, 256])
shape of tri_O: torch.Size([3, 32, 256])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
Max absolute values - ref: 0.1583251953125  tri: 0.1583251953125
Max absolute difference: 9.1552734375e-05
Benchmarking reference implementation...
Reference implementation: 7425.621 ms
Benchmarking Triton implementation...
Triton implementation: 0.467 ms
Speedup: 15890.855x
'''