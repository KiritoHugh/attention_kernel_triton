# Delivery Summary

core code files:
- magicpig_sparsetoken_decode_flash_attention_redundant_var_len_paged_with_norm.py
- sparsetoken_decode_flash_attention_redundant_var_len_paged.py

according log files:
- magicpig_paged_sparse_benchmark_20251204_192437.txt
- sparsetoken_decode_flash_attention_redundant_var_len_paged_20251203_191425.txt

# 1 attention_kernel_triton


## 1.1 Test on NVIDIA RTX 5000 Ada Generation

- `prefill_flash_attention.py`

Output:
```
>> Q: torch.Size([8, 64, 1024, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), causal: True, GQA_group_size: 4
Benchmarking reference implementation...
Reference implementation: 35.219 ms
Benchmarking Triton implementation...
Triton implementation: 0.780 ms
Speedup: 45.180x
```


- `prefill_flash_attention_var_len_padding.py`

Output:
```
>> Lengths: tensor([1024,  468,  631,  258,  353,  599,  732,   94], device='cuda:0')
>> Q: torch.Size([8, 64, 1024, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), causal: True, GQA_group_size: 4
Benchmarking reference implementation...
Reference implementation: 75.014 ms
Benchmarking Triton implementation...
Triton implementation: 0.596 ms
Speedup: 125.807x
```

- decode_flash_attention_redundant.py

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

- decode_flash_attention_redundant_var_len_paged.py

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

- sparsetoken_decode_flash_attention_redundant_var_len_paged.py

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

- sparsetoken_decode_flash_attention_redundant.py

Output:
```
>> q: torch.Size([1, 32, 1, 128]), K: torch.Size([1, 8, 32000, 128]), V: torch.Size([1, 8, 32000, 128]), GQA_group_size: 4
real kept ratio: 0.019923828125
shape of ref_O: torch.Size([1, 32, 1, 128])
shape of tri_O: torch.Size([1, 32, 1, 128])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
shape of ref_O_by_mask: torch.Size([1, 32, 1, 128])
Max absolute values - ref: 0.07025146484375  tri: 0.07025146484375
Max absolute difference: 3.0517578125e-05
Benchmarking reference implementation...
Reference implementation: 4.111 ms
Benchmarking naive_by_mask implementation...
Reference by mask implementation: 2.412 ms
Benchmarking Triton implementation...
Triton implementation: 0.057 ms
Speedup over reference: 71.995x
Speedup over reference by mask: 42.236x
```


# 2 Adaptation to HUAWEI Ascend 910B2


adptation bugs
1. [bug1] Change the order for K block_ptr from (0,1) to (1,0).



## 2.1 [DONE]

prefill_flash_attention.py [bug1 fixed]

prefill_flash_attention_var_len_padding.py [bug1 fixed]

decode_flash_attention_redundant.py [bug1 fixed]



## 2.2 [TODO]

decode_flash_attention_redundant_var_len_paged.py

- `for i in range(num_pages):` can not pass tirton compilation ...
    - hard code to `for i in range(1):` can pass test,
    - `for i in range(2):` can not pass compilation.
- try `for i in range(some tl.constexpr)` ? does not work either.

- 现在这个地方给我的感觉是，不管是for和if都用不了。 for的话只能for in range(1)，更大的数或者变量都会编译报错。if的话结果始终是错的，即使逻辑不改变，比如 i=0, if i<num_pages 结果算出的来是错的。

sparsetoken_decode_flash_attention_redundant.py

sparsetoken_decode_flash_attention_redundant_var_len_paged.py


## Test on HUAWEI Ascend 910B2 
- prefill_flash_attention.py

```
>> Q: torch.Size([8, 64, 1024, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), causal: True, GQA_group_size: 4
Benchmarking reference implementation...
Reference implementation: 25.782 ms
Benchmarking Triton implementation...
Triton implementation: 57.896 ms
Speedup: 0.445x
```

- prefill_flash_attention_var_len_padding.py

```
>> Lengths: tensor([1024,  875,   72, 1010,  212,  669,   61,  823], device='npu:0')
>> Q: torch.Size([8, 64, 1024, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), causal: True, GQA_group_size: 4
Benchmarking reference implementation...
Reference implementation: 25.727 ms
Benchmarking Triton implementation...
Triton implementation: 31.128 ms
Speedup: 0.826x
```

- decode_flash_attention_redundant.py

```
>> q: torch.Size([8, 64, 1, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), GQA_group_size: 4
 shape of ref: torch.Size([8, 64, 1, 64])
 shape of tri: torch.Size([8, 64, 1, 64])
Benchmarking reference implementation...
Reference implementation: 0.297 ms
Benchmarking Triton implementation...
Triton implementation: 1.469 ms
Speedup: 0.202x
```

- decode_flash_attention_redundant_var_len_paged.py

[triton ascend BUG] unsovable. see decode_flash_attention_redundant_var_len_paged.py:170-182
https://github.com/KiritoHugh/attention_kernel_triton/blob/92328eb49c37d1f5c936f0cea9f4e06f76433cf7/decode_flash_attention_redundant_var_len_paged.py

- sparsetoken_decode_flash_attention_redundant.py

```
(triton2) devserver-c001-5 attention_kernel_triton # python sparsetoken_decode_flash_attention_redundant.py 
>> q: torch.Size([1, 32, 1, 128]), K: torch.Size([1, 8, 32000, 128]), V: torch.Size([1, 8, 32000, 128]), GQA_group_size: 4
real kept ratio: 0.019853515625
shape of ref_O: torch.Size([1, 32, 1, 128])
Traceback (most recent call last):
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/triton_patch/compiler/compiler.py", line 284, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/backends/huawei/compiler.py", line 355, in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/backends/huawei/compiler.py", line 62, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/miniconda3/envs/triton2/lib/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/backends/huawei/triton-adapter-opt', '/tmp/tmpfscqniir/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpfscqniir/kernel.ttadapter.mlir']' died with <Signals.SIGSEGV: 11>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data1/dev1/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant.py", line 364, in <module>
    test_op_decode_sparsetoken(GQA_group_size=4, dtype=torch.float16)
  File "/data1/dev1/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant.py", line 314, in test_op_decode_sparsetoken
    tri_O = triton_implementation(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/dev1/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant.py", line 307, in triton_implementation
    return sparsetoken_flash_attention_decode(q, K, V, sparse_ind, sparse_nnz, softmax_scale, GQA_group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/dev1/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant.py", line 146, in sparsetoken_flash_attention_decode
    sparsetoken_flash_attention_decode_kernel[grid](
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/triton_patch/compiler/compiler.py", line 292, in compile
    raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
triton.triton_patch.compiler.errors.MLIRCompilationError: 
///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
<unknown>:0: warning: could not cast operand of type 'f16' to '!tt.ptr<f16>'
<unknown>:0: warning: could not cast operand of type 'f16' to '!tt.ptr<f16>'
<unknown>:0: warning: could not cast operand of type 'i32' to '!tt.ptr<i32>'
%206 = "tt.reshape"(%203) {MixUse} : (tensor<16xi32>) -> tensor<1x16xi32>
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/backends/huawei/triton-adapter-opt /tmp/tmpfscqniir/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpfscqniir/kernel.ttadapter.mlir
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-10-19-22:54:20 (PID:254890, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```


- sparsetoken_decode_flash_attention_redundant_var_len_paged.py

```
///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
<unknown>:0: warning: could not cast operand of type 'i32' to '!tt.ptr<i32>'
<unknown>:0: warning: could not cast operand of type 'f16' to '!tt.ptr<f16>'
<unknown>:0: warning: could not cast operand of type 'i32' to '!tt.ptr<i32>'
/data1/dev1/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant_var_len_paged.py:119:65: error: cannot div 0!
        page_id = tl.load(kv_page_indices_ptr + page_idx_start + page_idx, mask=mask, other=0)
                                                                ^
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /data1/miniconda3/envs/triton2/lib/python3.11/site-packages/triton/backends/huawei/triton-adapter-opt /tmp/tmpzo7t9wvp/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpzo7t9wvp/kernel.ttadapter.mlir
///------------------[ERROR][Triton][END]------------------
```
