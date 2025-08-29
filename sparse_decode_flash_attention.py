


def flash_attention_decode(q, K, V):
    pass

'''
By observing attnserver_dist.py, I found 
the "sparse attn" part is not using paged formate to store kv cache. and use mask to record locations. 

But I remember that the "sparse attn" in attnserver.py, which use cpu to compute, take in `ind` and `nnz` to do sparse attn. And its kv cache is not paged formate but single array formate, its kv cache is '沉默的大多数' part, it would not grow with decoding continuing. The decoding resulted kv cache would be only appended to the 'local token' part. 

For me, I think I have two options to handle this "sparse attn":
1. `ind` and `nnz` . but `ind` need to be padded to max seq len in batch. 
2. `inds` and `ptr` (and `nnz`). saving more index space. 

# Here, I retell the kv cache management and how decode on its kv cache, in MagicPIG dist.


'''
