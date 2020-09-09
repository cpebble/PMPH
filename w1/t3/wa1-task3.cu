#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 256
#endif
#ifndef N_ELEMS
  #define N_ELEMS 753411
#endif


// Src: Lab1-CudaIntro. Get time difference
int timeval_subtract( 
        struct timeval *result,
        struct timeval *t2,
        struct timeval *t1)
{
  unsigned int resolution = 1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - 
                  (t1->tv_usec + resolution * t2->tv_sec);
  result->tv_sec = diff / resolution; result->tv_usec = diff % resolution;
  return (diff<0);
}

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
  // Time tracking vars
  unsigned long int elapsed; 
  struct timeval t_start, t_end, t_diff;

  // Block distr vars
  unsigned int block_size = BLOCK_SIZE;
  unsigned int num_blocks = ((N + (block_size - 1)) / block_size);

  // Memory assignment
  unsigned int mem_size = N*sizeof(float);
  float* d_in;
  float* d_out;
  cudaMalloc((void**)&d_in, mem_size);
  cudaMalloc((void**)&d_out, mem_size);

  // Copy host mem to device
  cudaMemcpy(d_in, inp, mem_size, cudaMemcpyHostToDevice);
  // Exec kernel(with timetrack)
  gettimeofday(&t_start, NULL);
  kernel<<<num_blocks, block_size>>>(d_in, d_out, N);
  gettimeofday(&t_end, NULL);
  // Copy result from device to host
  cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);
  // Calculate and print time
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  printf("GPU Run took %d microseconds (%.2fms)\n", elapsed, elapsed / 1000.0);
}

void seq_run(float* inp, float* out, int N){
  unsigned long int elapsed; 
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);
  for(unsigned int i = 0; i < N; ++i){
    out[i] = pow(inp[i]/(inp[i]-2.3), 3);
  }
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  printf("CPU Run took %d microseconds (%.2fms)\n", elapsed, elapsed / 1000.0);
}

int main( int argc, char** argv){
  unsigned int N = N_ELEMS;
  unsigned int mem_size = N*sizeof(float);
  // Init memory arrays
  float* in = (float*) malloc(mem_size);
  float* gpu_out = (float*) malloc(mem_size);
  float* seq_out = (float*) malloc(mem_size);
  // And init the input array
  for (unsigned int i=0; i<N; ++i) in[i] = (float)i;

  // Run the code on the CPU
  seq_run(in, seq_out, N);
  // Run the code on the GPU
  gpu_run(in, gpu_out, N);

  // Now validate results:
  int passed = 0;
  int invalid = 0;
  for (int i = 0; i < N; ++i) {
    if (fabs(seq_out[i] - gpu_out[i]) < 0.0001)
        passed++;
    else invalid++;
  }
  printf("Passed: %06d, Invalid: %06d\n", passed, invalid);

  //DEBUG: Print the first 10 and last 10 values to 10p of precision
  // for(int i = 0; i < 10; i++) printf("%6d:\t%.10f\t%.10f\n", i, seq_out[i], gpu_out[i]);
  // for(int i = 0; i < 10; i++) printf("%6d:\t%.10f\t%.10f\n", N-i, seq_out[N-i], gpu_out[N-i]);
  // Free outpus databases
  free(in); free(gpu_out); free(seq_out);

  return 0;
}
