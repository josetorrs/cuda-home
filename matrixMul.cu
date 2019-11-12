#include <stdio.h>
#include <assert.h>

__global__ void matrixMulCUDA(float *A, float *B, float *C, int size){
    
    // Code from HW slide
    __shared__ float smem_c[64][64];
    __shared__ float smem_a[64][8];
    __shared__ float smem_b[8][64];

    int col = blockIdx.x * 64;
    int row = blockIdx.y * 64;

    for(int kk = 0; kk < size; kk += 8){
        for(int i = threadIdx.x + blockDim.x* threadIdx.y;
            i < 64 * 8; i += blockDim.x * blockDim.y){
            int k = kk + i / 64;
            int rt = row + i % 64;
            int ct = col + i % 64;
            smem_a[i%64][i/64] = A[rt*size+k];
            smem_b[i/64][i%64] = B[k*size+ct];
        }
    __syncthreads();
    // ....
    }

}

int main() {
    
    const int N = 8 * 64; // 256

    // Declare host memory for Matrices A and B
    float *hostA, *hostB, *hostC;

    // Declare device memory
    float *deviceA, *deviceB, *deviceC;

    // dimensions of the matrices
    const int aRows = 64;    // A = aRows * aCols 
    const int aCols = 8;     // B = bRows * bCols
    const int bRows = 8;
    const int bCols = 64;
    const int cRows = aRows; // C = aRows * bCols 
    const int cCols = bCols;

    // Allocate host memory for Matrices A and B
    size_t memSizeA = sizeof(float) * aRows * aCols;
    size_t memSizeB = sizeof(float) * bRows * bCols; 
    size_t memSizeC = sizeof(float) * cRows * cCols;
    float *hostA = reinterpret_cast<float* > (malloc(memSizeA));
    float *hostB = reinterpret_cast<float* > (malloc(memSizeB));
    float *hostC = reinterpret_cast<float* > (malloc(memSizeC));

    // sanity check
    assert(aRows == bCols);
    
    // init host memory

    // A = 2, 2, 2, 2 . . .
    // B = 3, 3, 3, 3 . . .
    for(int i=0; i < N; ++i){
        for(int j=0; j < N; ++j){
            hostA[i*N+j] = 2;
            hostB[i*N+j] = 3;
        }
    }

    // Allocate device memory
    cudaMalloc(reinterpret_cast<void **>(&deviceA), memSizeA);
    cudaMalloc(reinterpret_cast<void **>(&deviceB), memSizeB);
    cudaMalloc(reinterpret_cast<void **>(&deviceC), memSizeC);

    // Copy Matrix A and B from host to device
    cudaMemcpy(deviceA, hostA, memsizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, memsizeB, cudaMemcpyHostToDevice);

    // Init Kernel
    


    // Transfer from result matrix from device to host
    cudaMemcpy(hostC, deviceC, memsizeC, cudaMemcpyDeviceToHost);

    // free memory
    free(hostA);
    free(hostB);
    free(hostC);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return 0;
}
