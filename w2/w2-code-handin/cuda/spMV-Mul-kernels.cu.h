#ifndef SP_MV_MUL_KERS
#define SP_MV_MUL_KERS

__global__ void
replicate0(int tot_size, char* flags_d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < tot_size){
        flags_d[gid] = 0;
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0){
        printf("%d\n", mat_shp_sc_d[1]);
    }
    // ... fill in your implementation here ...
}

__global__ void 
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // ... fill in your implementation here ...
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    // ... fill in your implementation here ...
}

#endif
