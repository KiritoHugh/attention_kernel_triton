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


- `prefill_flash_attention_var_lenn_padding.py`

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
real kept ratio: 0.02015807622032198
shape of ref_O: torch.Size([3, 4, 64])
shape of tri_O: torch.Size([3, 4, 64])
Number of NaNs in triton_O: 0
Ratio of NaNs in triton_O: 0.0
Max absolute values - ref: 0.5751953125  tri: 0.5751953125
Max absolute difference: 0.00048828125
Benchmarking reference implementation...
Reference implementation: 59.383 ms
Benchmarking Triton implementation...
Triton implementation: 0.019 ms
Speedup: 3180.933x
```

