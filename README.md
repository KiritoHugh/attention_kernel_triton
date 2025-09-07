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
