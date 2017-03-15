# cuckoo
A GPU cuckoo lookup table based on CUDA

Just a simple lookup table on GPU. The hash table is based on cuckoo hash (https://en.wikipedia.org/wiki/Cuckoo_hashing). 

The building process is implemented on CPU, (with carefully choosen two hash functions (https://gist.github.com/badboy/6267743), usually, you don't need to rehash.)
>   create_hash_cpu(h_keys,h_values,h_key_1,h_key_2,h_value_1,h_value_2, N, EMPTY_KEY); 

The lookup process is on GPU (and I assume you won't insert or delete items later.)
> cuckoo_lookup<<<(M+255)/256, 256>>>(d_keys_lookup, d_values_lookup, d_key_1, d_value_1,  d_key_2, d_value_2, M, N, EMPTY_KEY, EMPTY_VALUE);
  
