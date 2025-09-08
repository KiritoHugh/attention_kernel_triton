import torch
import triton
import triton.language as tl
import os
q_redundant_len= 16



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
):
    seq_lo, seq_hi = 0, SEQ_LEN

    V_block_ptr = tl.advance(V_block_ptr, (seq_lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, seq_lo))

    # loop K V by KV_SEQ_BLOCK_SIZE
    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        # compute q@k
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

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
    q_ptr,
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
    GQA_group_size: tl.constexpr, 

    # block
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
):

    # divide data into blocks ans assign to its program.

    '''

    q : (B, H, 16, D)
    K : (B, H // G, S, D)
    V : (B, H // G, S, D)

    O = softmax(q @ K^T * scale) @ V

    The arrangement of workload:

    - The B and H dimension can be considered together to divide. (one program id) -> qkv_offset


    '''

    B_H_pid = tl.program_id(0)
    batch_id = B_H_pid // NUM_HEADS
    head_id = B_H_pid % NUM_HEADS

    qo_offset = batch_id * stride_Q_batch + head_id * stride_Q_head
    kv_offset = batch_id * stride_K_batch + (head_id // GQA_group_size) * stride_K_head 


    # O block
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        # 
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    # Q block
    Q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        # 
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        # 
        order = (1, 0),
    )

    # K block
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + kv_offset,
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
        base = V_ptr + kv_offset,
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
    - a q block
    - whole K,V 
    - a local O block and some local values
    '''

    # local O block and other local intermediate values
    tmp_O_block = tl.zeros((16, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((16,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((16,), dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # call programs 
    tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
        # query block
        Q_block,
        # 
        # K V ptrs
        K_block_ptr,
        V_block_ptr,
        # 
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
    )

    # tmp_m_i += tl.math.log(tmp_l_i)
    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    # m_ptrs = 
    # tl.store(m_ptrs, tmp_m_i)
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))


# flash_attention_decode
def flash_attention_decode(q, K, V, softmax_scale, GQA_group_size=1):

    q = q.repeat(1,1,16,1)
    VAR_KV_SEQ_BLK_SIZE = int(os.environ.get("VAR_KV_SEQ_BLK_SIZE", 64))

    # prepare some value for calling triton kernel
    B, H, S, D = K.shape
    assert q.shape == (B, H * GQA_group_size, 16, D)
    assert V.shape == (B, H , S, D)
    assert D % 64 == 0, "HEAD_DIM must be a multiple of 64?"

    # allocate output tensor
    O = torch.empty((B, H * GQA_group_size, 16, D), dtype=q.dtype, device=q.device)

    grid = (
        B * H * GQA_group_size,
        1,
        1,
    )

    flash_attention_kernel[grid](
        q,
        K,
        V,
        O,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BATCH_SIZE=B,
        NUM_HEADS=H * GQA_group_size,
        SEQ_LEN=S,
        HEAD_DIM=D,
        GQA_group_size=GQA_group_size,
        KV_SEQ_BLOCK_SIZE=VAR_KV_SEQ_BLK_SIZE,
    )

    return O[:,:,:1,:].contiguous()



def test_op_decode(BATCH_SIZE, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM, GQA_group_size = 1, dtype=torch.float16):


    q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS_KV*GQA_group_size, 1, HEAD_DIM),
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

    # print the shapes of Q K V
    print(f">> q: {q.shape}, K: {K.shape}, V: {V.shape}, GQA_group_size: {GQA_group_size}")

    softmax_scale = 1.0 / (HEAD_DIM**0.5)

    # reference implementation
    def reference_implementation(q, K, V, softmax_scale, GQA_group_size):
        if GQA_group_size > 1:
            K = K.repeat_interleave(GQA_group_size, dim=1)
            V = V.repeat_interleave(GQA_group_size, dim=1)
        P = torch.matmul(q, K.transpose(-2,-1)) * softmax_scale
        P = torch.softmax(P.float(), dim=-1).half()
        return torch.matmul(P, V)

    # triton implementation
    def triton_implementation(q, K, V, softmax_scale, GQA_group_size):
        return flash_attention_decode(q, K, V, softmax_scale, GQA_group_size)

    
    # # compare
    ref_O = reference_implementation(q, K, V, softmax_scale, GQA_group_size)
    tri_O = triton_implementation(q, K, V, softmax_scale, GQA_group_size)
    print(f" shape of ref: {ref_O.shape}")
    print(f" shape of tri: {tri_O.shape}")
    
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol), "The results are not close enough"

    
    # benchmark
    print("Benchmarking reference implementation...")
    ref_ms = triton.testing.do_bench(lambda: reference_implementation(q, K, V, softmax_scale, GQA_group_size))
    print(f"Reference implementation: {ref_ms:.3f} ms")

    print("Benchmarking Triton implementation...")
    tri_ms = triton.testing.do_bench(lambda: triton_implementation(q, K, V, softmax_scale, GQA_group_size))
    print(f"Triton implementation: {tri_ms:.3f} ms")

    print(f"Speedup: {ref_ms / tri_ms:.3f}x")


if __name__ == "__main__":

    # set VAR_Q_SEQ_BLK_SIZE, VAR_KV_SEQ_BLK_SIZE
    os.environ["VAR_KV_SEQ_BLK_SIZE"] = "64"


    test_op_decode(
        BATCH_SIZE=8,
        NUM_HEADS_KV=16,
        SEQ_LEN=1024,
        HEAD_DIM=64,
        GQA_group_size = 4,
    )


'''
Test on NVIDIA RTX 5000 Ada Generation

Output:
```
>> q: torch.Size([8, 64, 1, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), GQA_group_size: 4
 shape of ref: torch.Size([8, 64, 1, 64])
 shape of tri: torch.Size([8, 64, 1, 64])
Benchmarking reference implementation...
Reference implementation: 0.692 ms
Benchmarking Triton implementation...
Triton implementation: 0.102 ms
Speedup: 6.812x
```
'''
