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

# 16 = 16



@triton.jit
def aikg_custom_kernel(
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
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
    Triton 内核实现稀疏 token 分页注意力
    每个程序处理一个 (batch, head) 对
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
        # 如果 nnz 为 0，存储零输出
        tl.store(output_ptr + output_offset, 0.0)
        return
    
    # 加载查询向量 q [1, D]
    q_offset = b * H_q * D + h * D + tl.arange(0, D)
    q = tl.load(q_ptr + q_offset)
    
    # 计算 GQA 组和 KV 头索引
    groups = H_q // H_k
    h_k = h // groups
    
    # 初始化在线 softmax 状态
    m = -float('inf')
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    
    # 循环处理每个 token
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
        
        # 加载 K 向量 [1, D]
        k_offset = page_id * (2 * H_k * P * D) + 0 * (H_k * P * D) + h_k * (P * D) + offset_in_page * D + tl.arange(0, D)
        k = tl.load(paged_kv_cache_ptr + k_offset)
        
        # 加载 V 向量 [1, D]
        v_offset = page_id * (2 * H_k * P * D) + 1 * (H_k * P * D) + h_k * (P * D) + offset_in_page * D + tl.arange(0, D)
        v = tl.load(paged_kv_cache_ptr + v_offset)
        
        # 计算注意力分数
        dot_product = tl.sum(q * k)
        score = dot_product / sqrt_D  # 使用预计算的 sqrt_D
        
        # 更新在线 softmax
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(score - m_new)
        l_new = l * alpha + beta
        acc_new = acc * alpha + beta * v
        m = m_new
        l = l_new
        acc = acc_new
    
    # 计算最终输出
    out = acc / l
    tl.store(output_ptr + output_offset, out)
    

def sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):
        """
        Triton 稀疏 token 分页注意力
        """
        # 从输入张量获取形状参数
        B, H_q, _, D = q.shape
        M_pages, _, H_k, P, D_cache = paged_kv_cache.shape
        L_max = sparse_ind.shape[2]
        
        # 分配输出张量
        output = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)
        
        # 计算网格大小
        grid_size = B * H_q
        
        # 预计算 sqrt(D)
        sqrt_D = math.sqrt(D)
        
        # 启动内核
        aikg_custom_kernel[grid_size,](
            q,
            paged_kv_cache,
            kv_page_indptr,
            kv_page_indices,
            sparse_ind,
            sparse_nnz,
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


# HERE IS BASELINE 2: NPU Fusion Attention (Dense/Non-sparse) for Paged KV Cache
def sparsetoken_npu_fusion_attention_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):
    """
    使用 torch_npu.npu_fusion_attention 实现完全非稀疏的 attention (paged KV cache 版本)
    需要先将 paged KV cache 转换为连续格式
    
    q: [B, qo_heads, 1, head_dim]
    paged_kv_cache: [all_pages, 2, kv_heads, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    sparse_ind: [B, qo_heads, L_max]  (这个实现中不使用，为了接口统一)
    sparse_nnz: [B, qo_heads, 1]  (这个实现中不使用，为了接口统一)
    """
    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[3]
    num_kv_heads = paged_kv_cache.shape[2]
    page_size = paged_kv_cache.shape[3]
    groups = num_qo_heads // num_kv_heads
    
    k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]  # [pages, heads, page_size, head_dim]
    
    # 将 paged KV cache 转换为连续格式
    max_seq_len = 0
    K_list = []
    V_list = []
    
    for b in range(batch_size):
        num_pages = kv_page_indptr[b+1] - kv_page_indptr[b]
        seq_len = num_pages * page_size
        max_seq_len = max(max_seq_len, seq_len)
        
        # 为每个 batch 重建连续的 K 和 V
        batch_k_parts = []
        batch_v_parts = []
        for page_idx in range(num_pages):
            page_id = kv_page_indices[kv_page_indptr[b] + page_idx]
            batch_k_parts.append(k_cache[page_id])  # [kv_heads, page_size, head_dim]
            batch_v_parts.append(v_cache[page_id])
        
        batch_k = torch.cat(batch_k_parts, dim=1)  # [kv_heads, seq_len, head_dim]
        batch_v = torch.cat(batch_v_parts, dim=1)
        K_list.append(batch_k)
        V_list.append(batch_v)
    
    # Pad 到相同长度并堆叠
    K_padded_list = []
    V_padded_list = []
    for batch_k, batch_v in zip(K_list, V_list):
        seq_len = batch_k.shape[1]
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            batch_k = torch.nn.functional.pad(batch_k, (0, 0, 0, pad_len))
            batch_v = torch.nn.functional.pad(batch_v, (0, 0, 0, pad_len))
        K_padded_list.append(batch_k)
        V_padded_list.append(batch_v)
    
    K = torch.stack(K_padded_list, dim=0)  # [B, kv_heads, max_seq_len, head_dim]
    V = torch.stack(V_padded_list, dim=0)  # [B, kv_heads, max_seq_len, head_dim]
    
    # 如果是 GQA，需要将 K 和 V 扩展到与 q 相同的 head 数量
    if groups > 1:
        K = K.repeat_interleave(groups, dim=1)  # [B, qo_heads, max_seq_len, head_dim]
        V = V.repeat_interleave(groups, dim=1)  # [B, qo_heads, max_seq_len, head_dim]
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # 调用 npu_fusion_attention
    # sparse_mode=0 且不传入 atten_mask 表示不做 mask 操作（完全非稀疏）
    output, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        q,           # query: [B, qo_heads, 1, head_dim]
        K,           # key: [B, qo_heads, max_seq_len, head_dim]
        V,           # value: [B, qo_heads, max_seq_len, head_dim]
        head_num=num_qo_heads,
        input_layout="BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=None,  # 不传入 mask，表示完全非稀疏 attention
        scale=softmax_scale,
        keep_prob=1.0,
        pre_tockens=2147483647,  # 默认值
        next_tockens=2147483647,  # 默认值
        inner_precise=0,
        sparse_mode=0,  # defaultMask 模式，不传 mask 则不做 mask 操作
        gen_mask_parallel=True,
        sync=False
    )
    
    return output.squeeze(2)  # [B, qo_heads, head_dim]


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

    # random sparse, kept_ratio is passed as parameter
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
    real_kept_ratio = sparse_nnz.float().mean().item() / ((kv_page_indptr[1:] - kv_page_indptr[:-1]).float().mean().item() * page_size)
    log_print(f"real kept ratio: {real_kept_ratio}", output_file)



    # Create query
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)


    def triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz):
        return sparsetoken_flash_attention_decode_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)

    # compare 
    ref_O = sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)
    log_print(f"shape of ref_O: {ref_O.shape}", output_file)

    # just for evaluating the baseline in pytorch npu
    log_print("complete baseline running.", output_file)
    
    # Test NPU fusion attention (dense/non-sparse) for paged KV cache
    log_print("\n>> Testing NPU Fusion Attention (Dense) for Paged KV Cache...", output_file)
    npu_fusion_O = sparsetoken_npu_fusion_attention_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)
    log_print(f"shape of npu_fusion_O: {npu_fusion_O.shape}", output_file)
    
    # Test Triton implementation
    log_print("\n>> Testing Triton implementation...", output_file)
    tri_O = triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz)
    log_print(f"shape of tri_O: {tri_O.shape}", output_file)
    # triton_O how many nan? its ratio?
    log_print(f"Number of NaNs in triton_O: {torch.isnan(tri_O).sum().item()}", output_file)
    log_print(f"Ratio of NaNs in triton_O: {torch.isnan(tri_O).float().mean().item()}", output_file)


    # Correctness check for Triton implementation
    rtol = 0.0
    atol = 1e-2
    log_print("\n>> Correctness check for Triton implementation...", output_file)
    # print the max absolute values
    log_print(f"Max absolute values - ref: {torch.max(torch.abs(ref_O)).item()}, tri: {torch.max(torch.abs(tri_O)).item()}", output_file)
    # print the max absolute difference
    log_print(f"Max absolute difference: {torch.max(torch.abs(ref_O - tri_O)).item()}", output_file)
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"
    log_print("Triton implementation correctness check passed!", output_file)


    # benchmark
    log_print("\n" + "="*70, output_file)
    log_print("BENCHMARK RESULTS (Paged KV Cache)", output_file)
    log_print("="*70, output_file)
    
    log_print("Benchmarking Torch小算子拼接 (Reference - Paged)...", output_file)
    ref_ms = triton.testing.do_bench(lambda: sparsetoken_naive_paged_attention(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz))
    log_print(f"  Time: {ref_ms:.3f} ms", output_file)

    log_print("\nBenchmarking 融合全量Attention (NPU Fusion - Paged)...", output_file)
    npu_fusion_ms = triton.testing.do_bench(lambda: sparsetoken_npu_fusion_attention_paged(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz))
    log_print(f"  Time: {npu_fusion_ms:.3f} ms", output_file)

    log_print("\nBenchmarking Triton实现 (Sparse - Paged)...", output_file)
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz))
    log_print(f"  Time: {tri_ms:.3f} ms", output_file)

    log_print("\n" + "="*70, output_file)
    log_print("性能对比总结 (Paged KV Cache)", output_file)
    log_print("="*70, output_file)
    log_print(f"{'实现方式':<30} {'用时 (ms)':<15} {'相对加速':<15}", output_file)
    log_print("-"*70, output_file)
    log_print(f"{'Torch小算子拼接 (Paged)':<30} {ref_ms:>10.3f}    {'1.00x (baseline)':<15}", output_file)
    log_print(f"{'融合全量Attention (NPU)':<30} {npu_fusion_ms:>10.3f}    {ref_ms/npu_fusion_ms:>6.2f}x", output_file)
    log_print(f"{'Triton实现 (Sparse)':<30} {tri_ms:>10.3f}    {ref_ms/tri_ms:>6.2f}x", output_file)
    log_print("="*70, output_file)


def test_op_decode_paged_sparsetoken(GQA_group_size=4, dtype=torch.float16):
    """测试函数：使用多个 kept_ratio 进行测试"""
    import datetime
    
    # 创建输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"paged_sparse_attention_benchmark_{timestamp}.txt"
    
    kept_ratios = [0.02, 0.04, 0.05]
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        log_print("="*70, f)
        log_print("Paged Sparse Attention Benchmark - Multiple Ratios Test", f)
        log_print(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        log_print(f"GQA_group_size: {GQA_group_size}", f)
        log_print(f"dtype: {dtype}", f)
        log_print("="*70, f)
        
        for idx, kept_ratio in enumerate(kept_ratios, 1):
            log_print(f"\n\n{'#'*70}", f)
            log_print(f"第 {idx} 轮测试: kept_ratio = {kept_ratio}", f)
            log_print(f"{'#'*70}\n", f)
            
            test_op_decode_paged_sparsetoken_single_ratio(GQA_group_size, dtype, kept_ratio, f)
        
        log_print(f"\n\n{'='*70}", f)
        log_print("所有测试完成！", f)
        log_print(f"结果已保存到: {output_filename}", f)
        log_print("="*70, f)
    
    print(f"\n所有测试完成！结果已保存到: {output_filename}")


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
