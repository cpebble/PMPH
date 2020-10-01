#include <math.h>
#include <stdio.h>

void p_arr(float *arr,int n){
    int i = 0;
    printf("[");
    while (i < n){
        printf("%f ", arr[i++]);
    }
    printf("]\n");
}
int main() {
    int M = 10;
    int N = 5;
    float A[2*M];
    float A_[2*M];

    for (int i = 0; i < N; i++) {
        A[0] = N;

        for (int k = 1; k < 2*M; k++) {
            A[k] = sqrtf(A[k-1] * i * k);
        }

        //for (int j = 0; j < M; j++) {
            //B[i+1, j+1] = B[i, j] * A[2*j  ];
            //C[i,   j+1] = C[i, j] * A[2*j+1];
        //}
    }
    p_arr(A, 2*M);
    return 0;
}
