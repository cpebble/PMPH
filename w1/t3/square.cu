#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void kernel(float *d_in, float *d_out, int N){
  const unsigned int lid = threadIdx.x; // Local id inside a block
  const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
  if (gid < N){
    d_out[gid] = powf(d_in[gid]/(d_in[gid]-2.3), 3);
  }
}

int main( int argc, char** argv){
  unsigned int N = 753411;
  unsigned int mem_size = N*sizeof(float);
  unsigned int block_size = 256;
  unsigned int num_blocks = ((N + (block_size - 1)) / block_size);
  // Init host mem
  float* h_in = (float*) malloc(mem_size);
  float* h_out = (float*) malloc(mem_size);
  for (unsigned int i=0; i<N; ++i) h_in[i] = (float)i;

  // Init device mem
  float* d_in;
  float* d_out;
  cudaMalloc((void**)&d_in, mem_size);
  cudaMalloc((void**)&d_out, mem_size);

  // Copy host mem to device
  cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

  // Exec kernel
  kernel<<<num_blocks, block_size>>>(d_in, d_out, N);

  // Copy result from device to host
  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  // Print result
  for(unsigned int i=(N-10); i<N; ++i) printf("%d: %.6f\n", i, h_out[i]);

  free(h_in); free(h_out);
  cudaFree(d_in); cudaFree(d_out);

  return 0;
}
