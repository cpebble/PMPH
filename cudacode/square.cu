#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float *d_in, float *d_out){
  const unsigned int lid = threadIdx.x; // Local id inside a block
  const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
  d_out[gid] = d_in[gid]*d_in[gid]; // Place result(square) in d_out
}

int main( int argc, char** argv){
  unsigned int N = 32;
  unsigned int mem_size = N*sizeof(float);
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
  squarekernel<<< 1, N>>>(d_in, d_out);

  // Copy result from device to host
  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  // Print result
  for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

  free(h_in); free(h_out);
  cudaFree(d_in); cudaFree(d_out);

  return 0;
}
