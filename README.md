# attention_kernel_triton


Test on NVIDIA RTX 5000 Ada Generation

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


# Adaptation to HUAWEI Ascend 910B2


1. Change the order for K block_ptr from (0,1) to (1,0).
[DONE]
prefill_flash_attention.py
prefill_flash_attention_var_len_padding.py
decode_flash_attention_redundant.py


2. global varible not support so remove it.
[TODO]
decode_flash_attention_redundant_var_len_paged.py
- `for i in range(num_pages):` can not pass tirton compilation ...
 hard code to `for i in range(1):` can pass test,
 `for i in range(2):` can not pass compilation.
 try `for i in range(some tl.constexpr)` ?

- 现在这个地方给我的感觉是，不管是for和if都用不了。 for的话只能for in range(1)，更大的数或者变量都会编译报错。if的话结果始终是错的，即使逻辑不改变，比如 i=0, if i<num_pages 结果算出的来是错的。

sparsetoken_decode_flash_attention_redundant.py
sparsetoken_decode_flash_attention_redundant_var_len_paged.py