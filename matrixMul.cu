// NAME: Jose Torres
#include <stdio.h>
#include <iostream>
#include <assert.h>

__global__ void matrixMulCUDA(float *A, float *B, float *C, int size){

    // Code from HW slide
    __shared__ float smem_c[64][64];
    __shared__ float smem_a[64][8];
    __shared__ float smem_b[8][64];

    int col = blockIdx.x * 64;
    int row = blockIdx.y * 64;
    float val = 0;

    int start = threadIdx.x + blockDim.x * threadIdx.y;
    int stride = blockDim.x * blockDim.y;
    int end = 64 * 8;

    for(int kk = 0; kk < size; kk += 8){
        for(int i = start; i < end; i += stride){
            
            int k = kk + i / 64;
            int rt = row + i % 64;
            int ct = col + i % 64;
            smem_a[i%64][i/64] = A[rt*size+k];
            smem_b[i/64][i%64] = B[k*size+ct];
        
        }
        __syncthreads();
        // ....
        for(int k=kk; k < kk + 8; k++){
            // val += smem_a[threadIdx.x][k] * smem_b[k][threadIdx.y];
            smem_c[row][col] += smem_a[threadIdx.x][k] * smem_b[k][threadIdx.y];
        }
        __syncthreads();
    }
    // int i = col + threadIdx.x;
    // int j = row + threadIdx.y;
    // C[i*size+j] = val;
}

int main() {
    
    const int N = 8192; // something easily divisible by 256 (8*64)

    // Declare host memory for Matrices A and B
    float *hostA, *hostB, *hostC, *hostSumTemp;

    // Declare device memory
    float *deviceA, *deviceB, *deviceC;

    // Allocate host memory for all Matrices
    size_t memSize = sizeof(float) * N * N;
    hostA = reinterpret_cast<float* > (malloc(memSize));
    hostB = reinterpret_cast<float* > (malloc(memSize));
    hostC = reinterpret_cast<float* > (malloc(memSize));
    hostSumTemp = reinterpret_cast<float* > (malloc(memSize));
    
    // init host memory

    // A = 2, 2, 2, 2 . . .
    // B = 3, 3, 3, 3 . . .
    // C = 0, 0, 0, 0 . . .
    // hostSumTemp = 6, 6, 6, 6, ...
    // hostSumTemp is just temp variable to compare to
    // values are chosen to be really easy
    for(int i=0; i < N; ++i){
        for(int j=0; j < N; ++j){
            hostA[i*N+j] = 2;
            hostB[i*N+j] = 3;
            hostC[i*N+j] = 0;
            hostSumTemp[i*N*j] = 6;
        }
    }

    // Allocate device memory
    cudaMalloc(reinterpret_cast<void **>(&deviceA), memSize);
    cudaMalloc(reinterpret_cast<void **>(&deviceB), memSize);
    cudaMalloc(reinterpret_cast<void **>(&deviceC), memSize);

    // Copy Matrix A and B from host to device
    cudaMemcpy(deviceA, hostA, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, memSize, cudaMemcpyHostToDevice);

    // Invoke Kernel
    dim3 nblocks(N / 64, N / 64);
    dim3 nthreads(32, 32);

    // Init Kernel
    matrixMulCUDA<<< nblocks, nthreads>>> (hostA, hostB, hostC, N);
    
    // Transfer from result matrix from device to host
    cudaMemcpy(hostC, deviceC, memSize, cudaMemcpyDeviceToHost);

    // Check results
    bool isSame = true;
    int in, out;
    for(in=0; in < N; ++in){
        for(out=0; out < N; ++out){
            if(hostC[in*N*out] != hostSumTemp[in*N*out]){
                isSame = false;
                break;
            }
        }
    }

    // Print comparasion
    if(!isSame){
        std::cout << "Did not match at \n\ti = " << in << "\n\tj = " << out << std::endl;
    } else {
        std::cout << "Results matched.\n";
    }

    // free memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostSumTemp);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}