#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256


__global__ void kernel(float *d_in, float *d_out, int N){
  const unsigned int lid = threadIdx.x; // Local id inside a block
  const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
  if (gid < N){
    d_out[gid] = powf(d_in[gid]/(d_in[gid]-2.3), 3);
  }
}

void gpu_run(float* inp, float* out, int N)
{
  // Most of this code is stolened from the lab1 slides
  unsigned int block_size = BLOCK_SIZE;
  unsigned int num_blocks = ((N + (block_size - 1)) / block_size);
  float* d_in;
  float* d_out;
  unsigned_int mem_size = N*sizeof(float);
  // Cuda pointers calculated behind the scenes
  cudaMalloc((void**)&d_in, mem_size);
  cudaMalloc((void**)&d_out, mem_size);
  // Copy host mem to device
  cudaMemcpy(d_in, inp, mem_size, cudaMemcpyHostToDevice);
  // Exec kernel
  kernel<<<num_blocks, block_size>>>(d_in, d_out, N);
  // Copy result from device to host
  cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);
}

int main( int argc, char** argv){
  unsigned int N = 753411;
  unsigned int mem_size = N*sizeof(float);
  // Init memory arrays
  float* in = (float*) malloc(mem_size);
  float* gpu_out = (float*) malloc(mem_size);
  float* seq_out = (float*) malloc(mem_size);
  // And init the input array
  for (unsigned int i=0; i<N; ++i) h_in[i] = (float)i;

  // Run the code on the GPU
  gpu_run(in, gpu_out, N);
  // Free outpus databases
  free(in); free(gpu_out); free(seq_out);

  return 0;
}
