#include <iostream>
#include <cuda_runtime.h>



void set_to_val(int *data, int size, int val){
  for (int i = 0 ; i < size; i ++){
    data[i] = val;
  }
}

template<typename dType>
void print_matrix_gpu(dType *d_matrix,int rows,int cols) {
  dType * h_matrix = (dType *)malloc(rows*cols*sizeof(dType));
  cudaMemcpy(h_matrix, d_matrix, rows*cols*sizeof(dType), cudaMemcpyDeviceToHost);
  for(int i=0; i<rows; i++) {
    for(int j=0; j<cols; j++) {
      std::cout << h_matrix[i+j*rows] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  free(h_matrix);
}

template<typename dType>
void print_matrix(dType *h_matrix,int rows,int cols) {
  for(int i=0; i<rows; i++) {
    for(int j=0; j<cols; j++) {
      std::cout << h_matrix[i+j*rows] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}


__host__ __device__
int hash_func_1(int a){ // hash function on CPU, also need to have a on_device function
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

__host__ __device__
int hash_func_2(int key){ // hash function on CPU, also need to have a on_device function
  unsigned int c2=0x27d4eb2d; // a prime or an odd constant
  key = (key ^ 61) ^ (key >> 16);
  key = key + (key << 3);
  key = key ^ (key >> 4);
  key = key * c2;
  key = key ^ (key >> 15);
  return key;
}

template<typename vType>
void create_hash_cpu(int *h_keys, vType* h_values, int *h_key_1, int *h_key_2, int *h_value_1, int *h_value_2, int N,  int EMPTY_KEY){
  
  // create the cuckoo hash
  set_to_val(h_key_1, N , EMPTY_KEY);
  set_to_val(h_key_2, N , EMPTY_KEY);
  
  for (int i =0 ; i < N; i ++){
    int code = h_keys[i];
    vType value = h_values[i];
    
    // hash (key,value) into cuckoo
    int side = 0;
    while (true){
      if (side == 0){
	int key = (hash_func_1(code) % N + N) % N;
	if (h_key_1[key] == EMPTY_KEY){
	  h_key_1[key] = code;
	  h_value_1[key] = value;
	  break;
	} else {
	  int temp_code = h_key_1[key];
	  int temp_value = h_value_1[key];
	  h_key_1[key] = code;
	  h_value_1[key] = value;
	  code = temp_code;
	  value = temp_value;
	  side = 1;
	}
      } else {
	// side == 1
	int key = (hash_func_2(code) % N + N) % N;
	if (h_key_2[key] == -1){
	  h_key_2[key] = code;
	  h_value_2[key] = value;
	  break;
	} else {
	  int temp_code = h_key_2[key];
	  int temp_value = h_value_2[key];
	  h_key_2[key] = code;
	  h_value_2[key] = value;
	  code = temp_code;
	  value = temp_value;
	  side = 0;
	}
      }
    }
  }
}        


// M: how many keys to lookup, the size of d_keys;
// <<<(M+255)/256, 256>>>
template<typename dType>
__global__
void cuckoo_lookup(int *d_keys, dType *d_values,
                   int *d_key_1, dType *d_value_1, 
                   int *d_key_2, dType *d_value_2, 
                   int M, int N, int EMPTY_KEY, dType EMPTY_VALUE){
  for (int i = threadIdx.x; i < M; i += blockDim.x){
    int code = d_keys[i];
    //cuckoo lookup;
    int key1 = (hash_func_1(code) % N + N) % N;
    dType value = EMPTY_VALUE; 
    if (d_key_1[key1] == code){
      value = d_value_1[key1];
    } else {
      int key2 = (hash_func_2(code) % N + N) % N;
      if (d_key_2[key2] == code){
	value = d_value_2[key2];
      }
    }
    d_values[i] = value;
  }
}



template<typename dType>
void hdmalloc(dType **h_data, dType **d_data, int N){
  *h_data = (dType *)malloc(N*sizeof(dType));
  cudaMalloc((void **) d_data, N * sizeof(dType));
}

