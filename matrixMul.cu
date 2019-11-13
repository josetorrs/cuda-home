// NAME: Jose Torres
#include <stdio.h>
#include <iostream>
#include <assert.h>

__global__ void matrixMulCUDA(float *A, float *B, float *C, int size){

    // Code from HW slide
    __shared__ float smem_c[64][64];
    __shared__ float smem_a[64][8];
    __shared__ float smem_b[8][64];

    int c = blockIdx.x * 64;
    int r = blockIdx.y * 64;

    for(int kk = 0; kk < size; kk += 8){
        for(int i = threadIdx.x + blockDim.x * threadIdx.y;
                i < 64 * 8;
                i += blockDim.x * blockDim.y){
            
            // load into shared memory
            int k = kk + i / 64;
            int rt = r + i % 64;
            int ct = c + i % 64;
            smem_a[i%64][i/64] = A[rt*size+k];
            smem_b[i/64][i%64] = B[k*size+ct];
        
        }
        __syncthreads();
        for(int i=0; i < 2; ++i){
            for(int j=0; j < 2; ++j){

                int rowIdx = threadIdx.y * 2 + j;
                int colIdx = threadIdx.x * 2 + i;

                for(int k=0; k < 8; ++k){
                    // Store / Compute results in shared C
                    smem_c[rowIdx][colIdx] += smem_a[rowIdx][k] * smem_b[k][colIdx];
                }
                // Store back into global memory
                C[(r+rowIdx) * size + (c+colIdx)] = smem_c[rowIdx][colIdx];
            }
        }
    }
}

int main() {
    
    const int N = 8192; 

    // Declare host memory for Matrices A and B
    float *hostA, *hostB, *hostC, *hostSumTemp;

    // Declare device memory
    float *deviceA, *deviceB, *deviceC;

    // Allocate host memory for all Matrices
    size_t memSize = sizeof(float) * N * N;
    hostA = (float*) malloc(memSize);
    hostB = (float*) malloc(memSize);
    hostC = (float*) malloc(memSize);
    hostSumTemp = (float*) malloc(memSize);
    
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
            hostSumTemp[i*N+j] = 6;
        }
    }

    // Allocate device memory
    cudaMalloc((void**) &deviceA, memSize);
    cudaMalloc((void**) &deviceB, memSize);
    cudaMalloc((void**) &deviceC, memSize);

    // Copy Matrix A and B from host to device
    cudaMemcpy(deviceA, hostA, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, memSize, cudaMemcpyHostToDevice);

    // Invoke Kernel
    dim3 nblocks(N / 64, N / 64);
    dim3 nthreads(32, 32);

    // Init Kernel
    matrixMulCUDA<<<nblocks, nthreads>>> (deviceA, deviceB, deviceC, N);
    
    // Transfer from result matrix from device to host
    cudaMemcpy(hostC, deviceC, memSize, cudaMemcpyDeviceToHost);

    // Check results
    bool isSame = true;
    int row, col;
    for(row=0; row < N; ++row){
        for(col=0; col < N; ++col){
            if(hostC[row*N+col] != hostSumTemp[row*N+col]){
                isSame = false;
                break;
            }
        }
    }

    // Print comparasion
    if(!isSame){
        std::cout << "Did not match at \n\trow = " << row << "\n\tcol = " << col << std::endl;
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