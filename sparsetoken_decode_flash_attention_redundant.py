import torch
import triton
import triton.language as tl
import os

import math
redundant_len = 16



@triton.jit
def sparsetoken_flash_attention_decode_kernel(
    q_ptr, 
    K_ptr,
    V_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    O_ptr,
    softmax_scale,
    q_stride_B, q_stride_H, q_stride_1, q_stride_D,
    K_stride_B, K_stride_H, K_stride_S, K_stride_D,
    V_stride_B, V_stride_H, V_stride_S, V_stride_D,
    sparse_ind_stride_B, sparse_ind_stride_H, 
    sparse_nnz_stride_B, sparse_nnz_stride_H,
    O_stride_B, O_stride_H, O_stride_1, O_stride_D,
    # shapes
    B,
    num_qo_heads: tl.constexpr,
    head_dim: tl.constexpr,
    GQA_group_size: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
):
    '''
    q: (B, qo_heads, redundant_len, head_dim)
    K: (B, kv_heads, SEQ_LEN, head_dim)
    V: (B, kv_heads, SEQ_LEN, head_dim)
    sparse_ind: (B, qo_heads, L_max)  (padded with -1)
    sparse_nnz: (B, qo_heads)  (actual lengths)

    O = softmax(q @ K^T * scale) @ V
    '''

    B_H_pid = tl.program_id(0)
    out_batch_id = B_H_pid // num_qo_heads
    out_head_id = B_H_pid % num_qo_heads
    kv_head_id = out_head_id // GQA_group_size

    qo_offset = out_batch_id * q_stride_B + out_head_id * q_stride_H

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (redundant_len, head_dim),
        strides = (O_stride_1, O_stride_D),
        # 
        block_shape = (redundant_len, head_dim),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (redundant_len, head_dim),
        strides = (q_stride_1, q_stride_D),
        # 
        block_shape = (redundant_len, head_dim),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    q_block = tl.load(q_block_ptr)
    tmp_O_block = tl.zeros((redundant_len, head_dim), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    b_h_nzz = tl.load(sparse_nnz_ptr + out_batch_id * sparse_nnz_stride_B + out_head_id * sparse_nnz_stride_H)
    b_h_ind_ptr_base = sparse_ind_ptr + out_batch_id * sparse_ind_stride_B + out_head_id * sparse_ind_stride_H

    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    for ind_start_idx in range(0, b_h_nzz, KV_SEQ_BLOCK_SIZE):
        mask = ind_start_idx + KV_ranges < b_h_nzz
        token_idx = tl.load(b_h_ind_ptr_base + ind_start_idx + KV_ranges, mask=mask, other=0)
        k_ptr = K_ptr + out_batch_id * K_stride_B + kv_head_id * K_stride_H + token_idx * K_stride_S
        
        v_ptr = V_ptr + out_batch_id * V_stride_B + kv_head_id * V_stride_H + token_idx * V_stride_S

        shared_K = tl.load(
            k_ptr[None,:] + tl.arange(0, head_dim)[:, None] * K_stride_D,
            mask=mask[None, :],
            other=0.0
        )

        shared_V = tl.load(
            v_ptr[:,None] + tl.arange(0, head_dim)[None, :] * V_stride_D,
            mask=mask[:, None],
            other=0.0
        )

        # compute attention
        QK_block = tl.dot(q_block, shared_K)
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


def sparsetoken_flash_attention_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size):
    q = q.repeat_interleave(redundant_len, dim=2)  # [B, qo_heads, redundant_len, head_dim]
    B, num_qo_heads, _, head_dim = q.shape

    O = torch.zeros((B, num_qo_heads, redundant_len, head_dim), device=q.device, dtype=q.dtype)

    grid = (
        B * num_qo_heads,  # one block per (batch, head)
        1,
        1
    )
    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 16))

    sparsetoken_flash_attention_decode_kernel[grid](
        q,
        K,
        V,
        sparse_ind,
        sparse_nnz,
        O,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
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
        GQA_group_size,
        VAR_KV_SEQ_BLK_SIZE,
    )

    return O[:, :, :1, :].contiguous()  # [B, qo_heads, head_dim]


def sparsetoken_naive_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size):
    """
    q: [B, qo_heads, 1, head_dim]
    K: [B, kv_heads, SEQ_LEN, head_dim]
    V: [B, kv_heads, SEQ_LEN, head_dim]
    sparse_ind: [B, qo_heads, L_max]  (padded with -1)
    sparse_nnz: [B, qo_heads]  (actual lengths)
    """
    B, qo_heads, _, head_dim = q.shape
    kv_heads = K.shape[1]
    assert qo_heads % kv_heads == 0
    assert qo_heads // kv_heads == GQA_group_size
    device = q.device
    L_max = sparse_ind.shape[2]
    output = torch.zeros((B, qo_heads, 1, head_dim), dtype=q.dtype, device=device)
    for b in range(B):
        for h in range(qo_heads):
            k_h = h // GQA_group_size
            q_vec = q[b, h, 0]  # [head_dim]
            nnz = sparse_nnz[b, h].item()
            if nnz == 0:
                nnz = 1
            k_indices = sparse_ind[b, h, :nnz]  # [nnz]
            k_vecs = K[b, k_h, k_indices]  # [nnz, head_dim]
            v_vecs = V[b, k_h, k_indices]  # [nnz, head_dim]
            attn_scores = torch.matmul(k_vecs, q_vec) * softmax_scale  # [nnz]
            attn_probs = torch.softmax(attn_scores, dim=0)  # [nnz]
            out_vec = torch.matmul(attn_probs.unsqueeze(0), v_vecs)  # [1, head_dim]
            output[b, h, 0] = out_vec
    return output

def sparsetoken_naive_decode_by_mask(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size, mask):

    B, qo_heads, _, head_dim = q.shape
    _, kv_heads, SEQ_LEN, _ = K.shape
    assert qo_heads % kv_heads == 0
    assert qo_heads // kv_heads == GQA_group_size
    device = q.device
    L_max = sparse_ind.shape[2]

    # mask = torch.zeros((B, qo_heads, SEQ_LEN), dtype=torch.bool, device=device)
    # for b in range(B):
    #     for h in range(qo_heads):
    #         nnz = sparse_nnz[b, h].item()
    #         if nnz == 0:
    #             nnz = 1
    #         k_indices = sparse_ind[b, h, :nnz]  # [nnz]
    #         mask[b, h, k_indices] = True  # mark the positions to keep


    # then compute full attention, not use loop B, H
    K = K.repeat_interleave(GQA_group_size, dim=1)  # [B, qo_heads, SEQ_LEN, head_dim]
    V = V.repeat_interleave(GQA_group_size, dim=1)  # [B, qo_heads, SEQ_LEN, head_dim]
    attn_scores = torch.matmul(q, K.transpose(-2, -1)) * softmax_scale  # [B, qo_heads, 1, SEQ_LEN]
    attn_scores = attn_scores.masked_fill(~mask.unsqueeze(2), float('-inf'))  # mask out the unwanted positions
    attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, qo_heads, 1, SEQ_LEN]
    output = torch.matmul(attn_probs, V)  # [B, qo_heads, 1, head_dim]
    return output

    # then mask the output






def test_op_decode_sparsetoken(GQA_group_size = 4, dtype=torch.float16):
    pass

    device = "cuda"
    # Test parameters
    num_kv_heads = 8
    num_qo_heads = num_kv_heads * GQA_group_size
    head_dim = 256
    BATCH_SIZE = 4
    SEQ_LEN = 32000

    q = (
        torch.empty(
            (BATCH_SIZE, num_qo_heads, 1, head_dim),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )
    K = (
        torch.empty(
            (BATCH_SIZE, num_kv_heads, SEQ_LEN, head_dim),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )
    V = (
        torch.empty(
            (BATCH_SIZE, num_kv_heads, SEQ_LEN, head_dim),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )

    # print the shapes of Q K V
    print(f">> q: {q.shape}, K: {K.shape}, V: {V.shape}, GQA_group_size: {GQA_group_size}")

    softmax_scale = 1.0 / (head_dim**0.5)

    kept_ratio = 0.02
    L_max = int(SEQ_LEN * (kept_ratio+0.1))
    original_nzz = torch.zeros((BATCH_SIZE, num_qo_heads, 1), dtype=torch.int32, device=device)
    sparse_ind = torch.zeros((BATCH_SIZE, num_qo_heads, L_max), dtype=torch.int32, device=device) - 1
    # 
    original_nzz[:] = SEQ_LEN
    for bh in range(BATCH_SIZE*num_qo_heads):
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
    print("real kept ratio:", sparse_nnz.float().mean().item() / SEQ_LEN)


    ref_O = sparsetoken_naive_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)
    def triton_implementation(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size):
        return sparsetoken_flash_attention_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)



    # compare 
    ref_O = sparsetoken_naive_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)
    print("shape of ref_O:", ref_O.shape)
    tri_O = triton_implementation(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)
    print("shape of tri_O:", tri_O.shape)
    # triton_O how many nan? its ratio?
    print("Number of NaNs in triton_O:", torch.isnan(tri_O).sum().item())
    print("Ratio of NaNs in triton_O:", torch.isnan(tri_O).float().mean().item())


    precompute_mask = None
    # first construct mask from sparse_ind and sparse_nnz
    precompute_mask = torch.zeros((BATCH_SIZE, num_qo_heads, SEQ_LEN), dtype=torch.bool, device=device)
    for b in range(BATCH_SIZE):
        for h in range(num_qo_heads):
            nnz = sparse_nnz[b, h].item()
            if nnz == 0:
                nnz = 1
            k_indices = sparse_ind[b, h, :nnz]  # [nnz]
            precompute_mask[b, h, k_indices] = True  # mark the positions to keep

    ref_O_by_mask = sparsetoken_naive_decode_by_mask(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size, precompute_mask)
    print(f"shape of ref_O_by_mask: {ref_O_by_mask.shape}")
    assert torch.allclose(ref_O, ref_O_by_mask, atol=1e-2, rtol=0.0), "The results of naive and naive_by_mask are not close enough"


    rtol = 0.0
    atol = 1e-2
    # print the max absolute values
    print("Max absolute values - ref:", torch.max(torch.abs(ref_O)).item(), " tri:", torch.max(torch.abs(tri_O)).item())

    # print the max absolute difference
    print("Max absolute difference:", torch.max(torch.abs(ref_O - tri_O)).item())
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"


    # benchmark
    print("Benchmarking reference implementation...")
    ref_ms = triton.testing.do_bench(lambda: sparsetoken_naive_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    print("Benchmarking naive_by_mask implementation...")
    ref_by_mask_ms = triton.testing.do_bench(lambda: sparsetoken_naive_decode_by_mask(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size, precompute_mask))
    print(f"Reference by mask implementation: {ref_by_mask_ms:.3f} ms")

    print("Benchmarking Triton implementation...")
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size))
    print(f"Triton implementation: {tri_ms:.3f} ms")

    print(f"Speedup over reference: {ref_ms / tri_ms:.3f}x")
    print(f"Speedup over reference by mask: {ref_by_mask_ms / tri_ms:.3f}x")

if __name__ == "__main__":
    test_op_decode_sparsetoken(GQA_group_size=4, dtype=torch.float16)




'''
Test on NVIDIA RTX 5000 Ada Generation

Output:
```
>> q: torch.Size([4, 32, 1, 256]), K: torch.Size([4, 8, 32000, 256]), V: torch.Size([4, 8, 32000, 256]), GQA_group_size: 4
real kept ratio: 0.02011767578125
shape of ref_O: torch.Size([4, 32, 1, 256])
shape of tri_O: torch.Size([4, 32, 1, 256])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
shape of ref_O_by_mask: torch.Size([4, 32, 1, 256])
Max absolute values - ref: 0.0885009765625  tri: 0.0885009765625
Max absolute difference: 6.103515625e-05
Benchmarking reference implementation...
Reference implementation: 288.600 ms
Benchmarking naive_by_mask implementation...
Reference by mask implementation: 39.588 ms
Benchmarking Triton implementation...
Triton implementation: 0.462 ms
Speedup over reference: 624.374x
Speedup over reference by mask: 85.647x
```
'''
