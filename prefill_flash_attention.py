import torch
import triton
import triton.language as tl
import triton.testing
import os

'''
This kernel's case:
- The input Q K V has the same sequence length.
- They all starts from the first token.
- Support causal and non-causal attention.
- Only support forward for inference.
- Support bs and bs=1.
- Suppoer GQA

'''

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
    QO_SEQ_BLOCK_SIZE,
    KV_SEQ_BLOCK_SIZE,
    # 
    # block id
    seq_blk_id,
    # 
    # locals
    tmp_O_block,
    tmp_m_i,
    tmp_l_i,
    # 
    # stage
    STAGE,
    # 
    # ranges
    Q_ranges,
    KV_ranges,
    # 
    SEQ_LEN,
    # 
    # other
    softmax_scale,
):
    if STAGE == 1:
        # from 0 to the left of the diagonal; non-causal part in causal attn
        seq_lo, seq_hi = 0, seq_blk_id * QO_SEQ_BLOCK_SIZE
    elif STAGE == 2:
        # from the left of the diagonal to the remaining
        seq_lo, seq_hi = seq_blk_id * QO_SEQ_BLOCK_SIZE, (seq_blk_id + 1) * QO_SEQ_BLOCK_SIZE
        seq_lo = tl.multiple_of(seq_lo, QO_SEQ_BLOCK_SIZE)
    else:
        # for non-causal 
        seq_lo, seq_hi = 0, SEQ_LEN

    V_block_ptr = tl.advance(V_block_ptr, (seq_lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, seq_lo))

    # loop K V by KV_SEQ_BLOCK_SIZE
    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        # compute q@k
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # handle causal
        if STAGE == 2:
            mask = Q_ranges[:, None] >= KV_ranges[None, :] + start_kv
            QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)
            # mantain the max value 
            m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1) * softmax_scale )
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
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
def flash_attention_kernel(
    # data ptr
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    softmax_scale,

    # stride
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,

    # shapes
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    # block
    # 
    # Q_DIM_BLOCK_SIZE: tl.constexpr,
    # KV_DIM_BLOCK_SIZE: tl.constexpr,
    # QKV_DIM_BLOCK_SIZE: tl.constexpr,

    QO_SEQ_BLOCK_SIZE: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,

    # stage
    STAGE: tl.constexpr,

):
    pass
    tl.static_assert(QO_SEQ_BLOCK_SIZE <= HEAD_DIM)

    # divide data into blocks ans assign to its program.

    '''

    Q : (B, H, S, D)
    K : (B, H, S, D)
    V : (B, H, S, D)

    O = softmax(Q @ K^T * scale) @ V

    The arrangement of workload:

    - The B and H dimension can be considered together to divide. (one program id) -> qkv_offset
    - S dimension is divided. (one program id)


    '''

    B_H_pid = tl.program_id(0)
    batch_id = B_H_pid // NUM_HEADS
    head_id = B_H_pid % NUM_HEADS

    S_pid = tl.program_id(1)
    seq_blk_id = S_pid

    qkv_offset = batch_id * stride_Q_batch + head_id * stride_Q_head


    # O block
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        # 
        block_shape = (QO_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (seq_blk_id*QO_SEQ_BLOCK_SIZE, 0),
        # 
        order = (1, 0),
    )

    # Q block
    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        # 
        block_shape = (QO_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (seq_blk_id*QO_SEQ_BLOCK_SIZE, 0),
        # 
        order = (1, 0),
    )

    # K block
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + qkv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (
            stride_K_dim,
            stride_K_seq, 
            ),
        # 
        block_shape = (HEAD_DIM, KV_SEQ_BLOCK_SIZE),
        offsets = (0, 0),
        # 
        order = (0, 1),
    )

    # V block
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        # 
        block_shape = (KV_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )


    '''
    Then we need to define the actions in one program.

    One program needs:
    - a Q block
    - whole K,V 
    - a local Q block and some local values
    '''

    # local O block and other local intermediate values
    tmp_O_block = tl.zeros((QO_SEQ_BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((QO_SEQ_BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((QO_SEQ_BLOCK_SIZE,), dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # comput the ranges for threads in program
    Q_ranges = tl.arange(0, QO_SEQ_BLOCK_SIZE) + seq_blk_id * QO_SEQ_BLOCK_SIZE
    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)


    # call programs 
    if STAGE == 1 or STAGE == 3:
        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            # query block
            Q_block,
            # 
            # K V ptrs
            K_block_ptr,
            V_block_ptr,
            # 
            # block size
            QO_SEQ_BLOCK_SIZE,
            KV_SEQ_BLOCK_SIZE,
            # 
            # block id
            seq_blk_id,
            # 
            # locals
            tmp_O_block,
            tmp_m_i,
            tmp_l_i,
            # 
            # stage
            4 - STAGE,
            # 
            # ranges
            Q_ranges,
            KV_ranges,
            # 
            SEQ_LEN,
            # 
            # other
            softmax_scale,
        )

    if STAGE == 3:
        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            # query block
            Q_block,
            # 
            # K V ptrs
            K_block_ptr,
            V_block_ptr,
            # 
            # block size
            QO_SEQ_BLOCK_SIZE,
            KV_SEQ_BLOCK_SIZE,
            # 
            # block id
            seq_blk_id,
            # 
            # locals
            tmp_O_block,
            tmp_m_i,
            tmp_l_i,
            # 
            # stage
            2,
            # 
            # ranges
            Q_ranges,
            KV_ranges,
            # 
            SEQ_LEN,
            # 
            # other
            softmax_scale,
        )

    # tmp_m_i += tl.math.log(tmp_l_i)
    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    # m_ptrs = 
    # tl.store(m_ptrs, tmp_m_i)
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))





# flash_attention_
def flash_attention_(Q, K, V, causal, softmax_scale):

    VAR_Q_SEQ_BLK_SIZE = int(os.environ.get("VAR_Q_SEQ_BLK_SIZE", 64))
    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 64))

    # prepare some value for calling triton kernel
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert D % 64 == 0, "HEAD_DIM must be a multiple of 64?"

    # allocate output tensor
    O = torch.empty((B, H, S, D), dtype=Q.dtype, device=Q.device)
    stage = 3 if causal else 1

    grid = (
        B * H,
        triton.cdiv(S, VAR_Q_SEQ_BLK_SIZE),
        1,
    )

    flash_attention_kernel[grid](
        Q,
        K,
        V,
        O,
        softmax_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BATCH_SIZE=B,
        NUM_HEADS=H,
        SEQ_LEN=S,
        HEAD_DIM=D,
        QO_SEQ_BLOCK_SIZE=VAR_Q_SEQ_BLK_SIZE,
        KV_SEQ_BLOCK_SIZE=VAR_KV_SEQ_BLK_SIZE,
        STAGE=stage,
    )

    return O

def test_op(BATCH_SIZE, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    GQA_group_size = 1
    assert GQA_group_size == 1, "GQA not implemented yet"
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS_KV*GQA_group_size, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        ).normal_(mean=0, std=0.5)
    )

    softmax_scale = 1.0 / (HEAD_DIM**0.5)

    # reference implementation
    def reference_implementation(Q, K, V, causal, softmax_scale):
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda")) if causal else None
        P = torch.matmul(Q, K.transpose(-2,-1)) * softmax_scale
        if causal:
            if MASK is not None:
                P[:,:,MASK == 0] = float('-inf')
        P = torch.softmax(P.float(), dim=-1).half()
        return torch.matmul(P, V)

    # triton implementation
    def triton_implementation(Q, K, V, causal, softmax_scale):
        return flash_attention_(Q, K, V, causal, softmax_scale)

    # compare
    ref_O = reference_implementation(Q, K, V, causal, softmax_scale)
    tri_O = triton_implementation(Q, K, V, causal, softmax_scale)
    
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"

    # benchmark
    print("Benchmarking reference implementation...")
    ref_ms = triton.testing.do_bench(lambda: reference_implementation(Q, K, V, causal, softmax_scale))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    print("Benchmarking Triton implementation...")
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(Q, K, V, causal, softmax_scale))
    print(f"Triton implementation: {tri_ms:.3f} ms")

    print(f"Speedup: {ref_ms / tri_ms:.3f}x")


if __name__ == "__main__":

    # set VAR_Q_SEQ_BLK_SIZE, VAR_KV_SEQ_BLK_SIZE
    os.environ["VAR_Q_SEQ_BLK_SIZE"] = "64"
    os.environ["VAR_KV_SEQ_BLK_SIZE"] = "16"
    
    # When causal is true, KV_SEQ_BLK_SIZE must be Q_SEQ_BLK_SIZE/2^i where i>=0. 
    # Because when causal is true, the attn is done by two split parts, we need to ensure the dividing line should be dividable by KV_SEQ_BLK_SIZE.

    # When causal is false, this problem doesn't exist. 

    test_op(
        BATCH_SIZE=8,
        NUM_HEADS_KV=16,
        SEQ_LEN=1024,
        HEAD_DIM=64,
        causal=True,
    )


'''
Test on NVIDIA RTX 5000 Ada Generation

Output:
```
Benchmarking reference implementation...
Reference implementation: 8.889 ms
Benchmarking Triton implementation...
Triton implementation: 0.206 ms
Speedup: 43.220x
```
'''
