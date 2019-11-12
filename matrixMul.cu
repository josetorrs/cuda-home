#include <stdio.h>
#include <assert.h>


__global__ void matrixMulCUDA(float *A, float *B){
    
    // Block Index
    int xBlockIndex = blockIdx.x;
    int yBlockIndex = blockIdx.y;

    // Thread Index 
    int xThreadIndex = threadIdx.x;
    int yThreadIndex = threadIdx.y;

}

int main() {

    // Declare host memory for Matrices A and B
    float *hostA, *hostB, *hostC;

    // Declare device memory
    float *deviceA, *deviceB, *deviceC;

    // dimensions of the matrices
    const int dimsAx = 64;
    const int dimsAy = 8;
    const int dimsBx = 8;
    const int dimsBy = 64;

    // Allocate host memory for Matrices A and B
    size_t memSizeA = sizeof(float) * dimsAx * dimsAy;
    size_t memSizeB = sizeof(float) * dimsBx * dimsBy; 
    float *hostA = reinterpet_cast<float* > (malloc(memSizeA));
    float *hostB = reinterpet_cast<float* > (malloc(memSizeB));

    // init host memory

    // A = 2, 2, 2, 2 . . .
    for(int i=0; i < dimsAx; ++i){
        for(int j=0; i < dimsAy; ++j){
            A[i][j] = 2;
        }
    }


    // B = 3, 3, 3, 3 . . .
    for(int outIndex=0; outIndex < dimsBx; ++outIndex){
        for(int innerIndex=0; innerIndex < dimsBy; ++innerIndex){
            B[outIndex][innerIndex] = 3;
        }
    }

    // Allocate device memory
    cudaMalloc(reinterpret_cast<void **>(&deviceA), memSizeA);
    cudaMalloc(reinterpret_cast<void **>(&deviceB), memSizeB);


    // free memory
    free(hostA);
    free(hostB);
    free(hostC);

    cudaFree(deviceA);
    cudaFree(deviceB);
    

    return 0;
}
