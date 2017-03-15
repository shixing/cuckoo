#include "cuckoo.cuh"

int main(){
  int *h_keys, *h_values, *d_keys, *d_values;
  int *h_key_1, *h_key_2, *h_value_1, *h_value_2;
  int *d_key_1, *d_key_2, *d_value_1, *d_value_2;
  int N = 2048;

  int EMPTY_KEY = -1;
  int EMPTY_VALUE = -1;

  hdmalloc(&h_keys,&d_keys,N);
  hdmalloc(&h_values,&d_values,N);
  hdmalloc(&h_key_1,&d_key_1,N);
  hdmalloc(&h_key_2,&d_key_2,N);
  hdmalloc(&h_value_1,&d_value_1,N);
  hdmalloc(&h_value_2,&d_value_2,N);  

  //key = i , value = i*i;
  for (int i = 0 ; i < N; i++){
    h_keys[i] = i;
    h_values[i] = i*i; 
  }

  //create hash on cpu 
  create_hash_cpu(h_keys,h_values,h_key_1,h_key_2,h_value_1,h_value_2, N, EMPTY_KEY); 
 
  // copy to GPU
  cudaMemcpy(d_key_1, h_key_1, N * sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_key_2, h_key_2, N * sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_value_1, h_value_1, N * sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_value_2, h_value_2, N * sizeof(int),cudaMemcpyHostToDevice);

  //lookup on GPU
  int M = 300;
  int *h_keys_lookup, *h_values_lookup, *d_keys_lookup, *d_values_lookup;
  hdmalloc(&h_keys_lookup,&d_keys_lookup,M);
  hdmalloc(&h_values_lookup,&d_values_lookup,M);

  for (int i = 0 ; i < M; i ++){
    h_keys_lookup[i] = 2*i;
  }
  cudaMemcpy(d_keys_lookup, h_keys_lookup, M * sizeof(int),cudaMemcpyHostToDevice);
  
  
  cuckoo_lookup<<<(M+255)/256, 256>>>(d_keys_lookup, d_values_lookup,
				      d_key_1, d_value_1, 
				      d_key_2, d_value_2, 
				      M, N, EMPTY_KEY, EMPTY_VALUE);
  
  std::cout << "keys:\n";
  print_matrix_gpu(d_keys_lookup,1,M);
  std::cout << "values:\n";
  print_matrix_gpu(d_values_lookup,1,M);

}
