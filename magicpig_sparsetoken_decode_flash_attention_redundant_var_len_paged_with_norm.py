import torch
import torch_npu
DEVICE_STR = "npu"
import triton
import triton.language as tl
import os
import math

def log_print(msg, output_file=None):
    """同时打印到控制台和文件"""
    print(msg)
    if output_file:
        output_file.write(msg + "\n")
        output_file.flush()

@triton.jit
def magicpig_custom_kernel(
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    K_ptr,              # MagicPig 参数 K
    L_ptr,              # MagicPig 参数 L
    output_ptr,
    B: tl.constexpr,
    H_q: tl.constexpr,
    H_k: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    L_max: tl.constexpr,
    sqrt_D: tl.constexpr,
):
    """
    Triton 内核实现稀疏 token 分页注意力 (MagicPig Version)
    风格：Code 2 (逐 Token 循环，手动指针计算)
    逻辑：Code 1 (MagicPig Norm + Attention)
    """
    pid = tl.program_id(0)
    b = pid // H_q
    h = pid % H_q
    
    # 加载 nnz
    offset_nnz = b * H_q + h
    nnz = tl.load(sparse_nnz_ptr + offset_nnz).to(tl.int32)
    
    # 计算输出偏移
    output_offset = b * H_q * D + h * D + tl.arange(0, D)
    
    if nnz == 0:
        tl.store(output_ptr + output_offset, 0.0)
        return
    
    # 加载查询向量 q [1, D]
    q_offset = b * H_q * D + h * D + tl.arange(0, D)
    q = tl.load(q_ptr + q_offset)
    
    # MagicPig: 计算 Q 的模长 (需转为 float32 计算以保证精度)
    q_f32 = q.to(tl.float32)
    q_norm = tl.sqrt(tl.sum(q_f32 * q_f32))

    # MagicPig: 加载当前 Head 的 K, L 参数
    # 假设 K, L 为 [B, H_q] 且连续
    param_offset = b * H_q + h
    K_val = tl.load(K_ptr + param_offset)
    L_val = tl.load(L_ptr + param_offset)

    # 计算 GQA 组和 KV 头索引
    groups = H_q // H_k
    h_k = h // groups
    
    # 初始化在线 softmax 状态
    m = -float('inf')
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    
    PI = 3.14159265358979323846

    # 循环处理每个 token (Code 2 风格循环)
    for i in range(0, nnz):
        # 加载 token 索引
        offset_ind = b * H_q * L_max + h * L_max + i
        token_idx = tl.load(sparse_ind_ptr + offset_ind).to(tl.int32)
        
        # 计算页面索引和偏移
        page_idx = token_idx // P
        offset_in_page = token_idx % P
        
        # 加载页面起始指针
        ptr_start = tl.load(kv_page_indptr_ptr + b).to(tl.int32)
        
        # 加载页面 ID
        page_id_offset = ptr_start + page_idx
        page_id = tl.load(kv_page_indices_ptr + page_id_offset).to(tl.int32)
        
        # 计算 KV 指针偏移
        # paged_kv_cache: [all_pages, 2, kv_heads, page_size, head_dim]
        # 展平偏移计算
        base_offset = page_id * (2 * H_k * P * D) + h_k * (P * D) + offset_in_page * D
        k_offset = base_offset + 0 * (H_k * P * D) + tl.arange(0, D)
        v_offset = base_offset + 1 * (H_k * P * D) + tl.arange(0, D)

        # 加载 K, V 向量
        k_vec = tl.load(paged_kv_cache_ptr + k_offset)
        v_vec = tl.load(paged_kv_cache_ptr + v_offset)
        
        # --- MagicPig Logic Start ---
        
        # 1. 计算原始 Dot Product
        dot_product = tl.sum(q * k_vec)
        
        # 2. 计算 K 的模长
        k_vec_f32 = k_vec.to(tl.float32)
        k_norm = tl.sqrt(tl.sum(k_vec_f32 * k_vec_f32))
        
        # 3. 计算归一化的 qk (Cosine Similarity 类似物)
        # qk_normalized = qk^T / (|q| * |k|)
        denom = q_norm * k_norm + 1e-8
        qk_normalized = dot_product / denom
        
        # Clamp 避免 acos 出错
        qk_normalized = tl.where(qk_normalized > 1.0 - 1e-7, 1.0 - 1e-7, qk_normalized)
        qk_normalized = tl.where(qk_normalized < -1.0 + 1e-7, -1.0 + 1e-7, qk_normalized)
        
        # 4. 计算 p_i = 1 - (1/π) * arccos(qk_normalized)
        p_i = 1.0 - (1.0 / PI) * tl.math.acos(qk_normalized)
        p_i = tl.where(p_i > 1.0, 1.0, p_i)
        p_i = tl.where(p_i < 0.0, 0.0, p_i)
        
        # 5. 计算 u_i (MagicPig 衰减系数)
        # u_i = 1 - (1 - p_i^K)^L - L*p_i^K * (1 - p_i^K)^(L-1)
        p_i_K = tl.math.pow(p_i, K_val)
        one_minus_p_i_K = 1.0 - p_i_K
        one_minus_p_i_K = tl.where(one_minus_p_i_K > 1.0, 1.0, one_minus_p_i_K)
        one_minus_p_i_K = tl.where(one_minus_p_i_K < 0.0, 0.0, one_minus_p_i_K)
        
        one_minus_p_i_K_L = tl.math.pow(one_minus_p_i_K, L_val)
        one_minus_p_i_K_L_minus_1 = tl.math.pow(one_minus_p_i_K, L_val - 1.0)
        
        u_i = 1.0 - one_minus_p_i_K_L - L_val * p_i_K * one_minus_p_i_K_L_minus_1
        u_i = tl.where(u_i <= 0.0, 1e-10, u_i)
        
        # 6. 计算 log(u_i) 并调整 score
        log_u_i = tl.math.log(u_i)
        
        # Final Score: exp(qk^T/sqrt(d) - log(u_i)) -> Score = qk^T/sqrt(d) - log(u_i)
        score = (dot_product / sqrt_D) - log_u_i
        
        # --- MagicPig Logic End ---
        
        # 更新在线 softmax
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(score - m_new)
        l_new = l * alpha + beta
        acc_new = acc * alpha + beta * v_vec
        m = m_new
        l = l_new
        acc = acc_new
    
    # 计算最终输出
    out = acc / l
    tl.store(output_ptr + output_offset, out)

def magicpig_sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):
    """
    MagicPig 稀疏 Token 分页 Attention 封装 (Code 2 风格)
    """
    # 从输入张量获取形状参数
    B, H_q, _, D = q.shape
    # paged_kv_cache: [M_pages, 2, H_k, P, D]
    M_pages, _, H_k, P, D_cache = paged_kv_cache.shape
    L_max = sparse_ind.shape[2]
    
    # 确保 K, L 是连续的 [B, H_q]
    K = K.contiguous()
    L = L.contiguous()

    # 分配输出张量
    output = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
    
    # 计算网格大小
    grid_size = B * H_q
    
    # 预计算 sqrt(D)
    sqrt_D = math.sqrt(D)
    
    # 启动内核
    magicpig_custom_kernel[grid_size,](
        q,
        paged_kv_cache,
        kv_page_indptr,
        kv_page_indices,
        sparse_ind,
        sparse_nnz,
        K,
        L,
        output,
        B,
        H_q,
        H_k,
        D,
        P,
        L_max,
        sqrt_D,
    )
    
    return output

# HERE IS THE BASELINE (From Code 1)
def magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):
    '''
    q: [B, qo_heads, 1, head_dim]
    paged_kv_cache: [B, 2, H, page_size, head_dim] (Actually [total_pages, ...])
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
                batch_outputs.append(torch.zeros((1, head_dim), device=q.device, dtype=q.dtype))
                continue
            
            seq_head_k_parts = []
            seq_head_v_parts = []
            indices = sparse_ind[b, h, :nnz]
            kv_h = h // groups
            for i in range(nnz):
                token_idx = indices[i].item()
                page_idx = token_idx // page_size
                page_id = kv_page_indices[kv_page_indptr[b].item() + page_idx]
                offset_in_page = token_idx % page_size
                seq_head_k_parts.append(k_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])
                seq_head_v_parts.append(v_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])
            seq_head_k = torch.cat(seq_head_k_parts, dim=0).unsqueeze(0)  # [1, seq_len, head_dim]
            seq_head_v = torch.cat(seq_head_v_parts, dim=0).unsqueeze(0)

            K_val = K[b, h].item()
            L_val = L[b, h].item()

            q_bh = q[b, h:h+1].float()
            seq_head_k_f32 = seq_head_k.float()
            
            # Compute QK scores
            scores = torch.matmul(q_bh, seq_head_k_f32.permute(0, 2, 1))  # [1, 1, seq_len]
            
            # Compute norms
            q_norm = torch.norm(q_bh, dim=-1, keepdim=True)
            k_norm = torch.norm(seq_head_k_f32, dim=-1, keepdim=True)
            
            # Compute normalized dot product
            qk_normalized = scores / (q_norm * k_norm.permute(0, 2, 1) + 1e-8)
            qk_normalized = torch.clamp(qk_normalized, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # MagicPig Math
            p_i = 1.0 - (1.0 / math.pi) * torch.acos(qk_normalized)
            p_i = torch.clamp(p_i, 0.0, 1.0)
            
            p_i_K = torch.pow(p_i, K_val)
            one_minus_p_i_K = 1.0 - p_i_K
            one_minus_p_i_K = torch.clamp(one_minus_p_i_K, 0.0, 1.0)
            
            one_minus_p_i_K_L = torch.pow(one_minus_p_i_K, L_val)
            one_minus_p_i_K_L_minus_1 = torch.pow(one_minus_p_i_K, L_val - 1.0)
            
            u_i = 1.0 - one_minus_p_i_K_L - L_val * p_i_K * one_minus_p_i_K_L_minus_1
            u_i = torch.clamp(u_i, min=1e-10)
            
            log_u_i = torch.log(u_i)
            
            # Apply attention formula
            scores = scores / math.sqrt(head_dim) - log_u_i
            
            weights = torch.softmax(scores, dim=-1)
            seq_head_v_f32 = seq_head_v.float()
            out = torch.matmul(weights, seq_head_v_f32)
            batch_outputs.append(out.squeeze(1).to(q.dtype))
        outputs.append(torch.cat(batch_outputs, dim=0).unsqueeze(0))

    return torch.cat(outputs, dim=0)

def test_op_decode_paged_sparsetoken_single_ratio(GQA_group_size, dtype, kept_ratio, output_file=None):
    """单次测试，使用指定的 kept_ratio"""
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
    
    # Create metadata
    batch_size = 3
    num_pages_per_seq = torch.tensor([270, 350, 321], dtype=torch.int32, device=device)
    kv_page_indices = torch.tensor(range(sum(num_pages_per_seq)), dtype=torch.int32, device=device)
    kv_page_indices = kv_page_indices[torch.randperm(kv_page_indices.shape[0], device=device)]
    kv_page_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
    kv_page_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    kv_last_page_len = torch.tensor([63, 22, 55], dtype=torch.int32, device=device)
    if torch.any(kv_last_page_len > page_size):
        kv_last_page_len.fill_(page_size-1)

    # random sparse
    L_max = max_num_pages_per_seq * page_size
    original_nzz = torch.zeros((batch_size, num_qo_heads, 1), dtype=torch.int32, device=device)
    sparse_ind = torch.zeros((batch_size, num_qo_heads, L_max), dtype=torch.int32, device=device) - 1
    
    for b in range(batch_size):
        original_nzz[b] = (kv_page_indptr[b+1] - kv_page_indptr[b]) * page_size - (page_size - kv_last_page_len[b])

    for bh in range(batch_size*num_qo_heads):
        b = bh // num_qo_heads
        h = bh % num_qo_heads
        sample_prob = torch.rand((original_nzz[b][h].item(),), device=device)
        kept_mask = sample_prob < kept_ratio
        kept_nnz = kept_mask.sum().item()
        if kept_nnz == 0:
            kept_nnz = 1
            kept_mask[0] = True
        original_nzz[b][h] = kept_nnz
        sparse_ind[b, h, :kept_nnz] = torch.nonzero(kept_mask, as_tuple=False).squeeze(-1)
    sparse_nnz = original_nzz
    
    real_kept_ratio = sparse_nnz.float().mean().item() / ((kv_page_indptr[1:] - kv_page_indptr[:-1]).float().mean().item() * page_size)
    log_print(f"real kept ratio: {real_kept_ratio}", output_file)

    # MagicPig 参数 K, L
    K = torch.rand(batch_size, num_qo_heads, device=device, dtype=torch.float32) * 5 + 1
    L = torch.rand(batch_size, num_qo_heads, device=device, dtype=torch.float32) * 5 + 1

    # Create query
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)

    def triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L):
        return magicpig_sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)

    # compare 
    ref_O = magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)
    log_print(f"shape of ref_O: {ref_O.shape}", output_file)
    log_print("complete baseline running.", output_file)
    
    # Test Triton implementation
    log_print("\n>> Testing Triton implementation (MagicPig)...", output_file)
    tri_O = triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L)
    log_print(f"shape of tri_O: {tri_O.shape}", output_file)
    log_print(f"Number of NaNs in triton_O: {torch.isnan(tri_O).sum().item()}", output_file)
    
    # Correctness check
    rtol = 0.0
    atol = 1e-2
    log_print("\n>> Correctness check...", output_file)
    log_print(f"Max absolute values - ref: {torch.max(torch.abs(ref_O)).item()}, tri: {torch.max(torch.abs(tri_O)).item()}", output_file)
    diff = torch.abs(ref_O - tri_O)
    log_print(f"Max absolute difference: {torch.max(diff).item()}", output_file)
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"
    log_print("Triton implementation correctness check passed!", output_file)

    # benchmark
    log_print("\n" + "="*70, output_file)
    log_print("BENCHMARK RESULTS (MagicPig Paged)", output_file)
    log_print("="*70, output_file)
    
    log_print("Benchmarking Reference (PyTorch Loop)...", output_file)
    ref_ms = triton.testing.do_bench(lambda: magicpig_sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L))
    log_print(f"  Time: {ref_ms:.3f} ms", output_file)

    log_print("\nBenchmarking Triton (MagicPig)...", output_file)
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz, K, L))
    log_print(f"  Time: {tri_ms:.3f} ms", output_file)

    log_print("\n" + "="*70, output_file)
    log_print("性能对比", output_file)
    log_print("-" * 70, output_file)
    log_print(f"{'Reference':<30} {ref_ms:>10.3f} ms", output_file)
    log_print(f"{'Triton (MagicPig)':<30} {tri_ms:>10.3f} ms    Speedup: {ref_ms/tri_ms:>6.2f}x", output_file)
    log_print("="*70, output_file)

def test_op_decode_paged_sparsetoken(GQA_group_size=4, dtype=torch.float16):
    """测试函数：使用多个 kept_ratio 进行测试"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"magicpig_paged_sparse_benchmark_{timestamp}.txt"
    
    kept_ratios = [0.02, 0.04]
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        log_print("="*70, f)
        log_print("MagicPig Paged Sparse Attention Benchmark", f)
        log_print(f"GQA_group_size: {GQA_group_size}", f)
        log_print(f"dtype: {dtype}", f)
        log_print("="*70, f)
        
        for idx, kept_ratio in enumerate(kept_ratios, 1):
            log_print(f"\n\n{'#'*70}", f)
            log_print(f"Round {idx}: kept_ratio = {kept_ratio}", f)
            log_print(f"{'#'*70}\n", f)
            test_op_decode_paged_sparsetoken_single_ratio(GQA_group_size, dtype, kept_ratio, f)
        
        log_print(f"\nDone. Results saved to {output_filename}", f)
    
    print(f"\nDone. Results saved to {output_filename}")

if __name__ == "__main__":
    test_op_decode_paged_sparsetoken(GQA_group_size=4, dtype=torch.float16)
