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
>> Q: torch.Size([8, 64, 1024, 64]), K: torch.Size([8, 16, 1024, 64]), V: torch.Size([8, 16, 1024, 64]), causal: True, GQA_group_size: 4
Benchmarking reference implementation...
Reference implementation: 74.974 ms
Benchmarking Triton implementation...
Triton implementation: 0.868 ms
Speedup: 86.400x
```
