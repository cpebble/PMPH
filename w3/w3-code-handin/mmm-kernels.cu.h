#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void matMultKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  __shared__ ElTp Ash[T][T];
  __shared__ ElTp Bsh[T][T];

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      Ash[threadIdx.y][threadIdx.x] = ((gidy < heightA) && (kk+threadIdx.x < widthA)) ?
            A[gidy*widthA + kk + threadIdx.x] : 0.0;
      Bsh[threadIdx.y][threadIdx.x] = ((gidx < widthB)  && (kk+threadIdx.y < widthA)) ?
            B[(threadIdx.y+kk)*widthB + gidx] : 0.0;
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < T; k++)
          accum += Ash[threadIdx.y][k] * Bsh[k][threadIdx.x];
      __syncthreads();
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}

// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultCacheKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      __syncthreads();
      #pragma unroll
      for(int k = 0; k < T; k++)
        accum += A[gidy*widthA + kk + k] * B[gidy*widthB + (kk+k)];
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}


template <class ElTp, int T> 
__global__ void matMultRegTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    // ToDo: fill in the kernel implementation of register+block tiled 
    //       matrix-matrix multiplication here
    // These calculations are wrong
    int ii = blockIdx.y * T;
    int j__= blockIdx.x * T * T;
    int i; // Altid sequential
    int j_ = (threadIdx.y*T) + j__;
    int j  = j_ + threadIdx.x;
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int tidy = threadIdx.y, tidx = threadIdx.x;
    __shared__ float Ash[T][T];
    ElTp cs[T];
    // printf("%d: %d \n", j, gidy);
    #pragma unroll
    for (int i = 0; i < T; i++){
        cs[i] = 0;
    }
    __syncthreads();
    // So now we add the sequential K loop that actually does "something"
    for(int kk = 0; kk < widthA; kk +=T ){
        // Copy the array slice A[ii:ii+T, j] into shared memory
        if ((kk+tidx) < widthB && gidy < heightA)
            Ash[tidy][tidx] = A[(kk+tidx) + (gidy*widthA)];
        else
            Ash[tidy][tidx] = 0.0f;

        __syncthreads();
        // Then synchronize
        for (int k = 0; k < T; k++) {
            float b;
            if (kk+k < widthA && j < widthB)
                b = B[(kk+k)*widthB+(j)];
            else
                b = 0.0f;
            #pragma unroll
            for(int i = 0; i < T; i++){
                cs[i] += Ash[i][k] * b;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < T; i++){
        if ((ii + i < heightA) && (j < widthB)){
            C[((ii + i) * widthB) + j] = cs[i];
        }
    }

}


#endif
