#include <stdio.h>
int N = 3;

void p_arr(float *arr,int n){
    int i = 0;
    printf("[");
    while (i < n){
        printf("%f ", arr[i++]);
    }
    printf("]\n");
}

int main(){
    float A[N][64];
    float B[N][64];
    for(int i = 0; i < 64; i++){
        A[0][i] = (float) i;
        A[1][i] = (float) i;
        A[2][i] = (float) i;
    }
    p_arr(A[0], 64);
    p_arr(A[1], 64);
    p_arr(A[2], 64);
    float accum, tmpA;
    for (int i = 0; i < N; i++) { // outer loop
        accum = 0;
        for (int j = 0; j < 64; j++) { // inner loop
            tmpA = A[i][j];
            accum = accum + tmpA*tmpA; // (**)
            B[i][j] = accum;
        }
    }
    p_arr(B[0], 64);
    p_arr(B[1], 64);
    p_arr(B[2], 64);

}

