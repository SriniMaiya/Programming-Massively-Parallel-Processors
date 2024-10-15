#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(const float *A, const float *B, float *C, const int M, const int K, const int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < M && col < N)
    {
        for (int l = 0; l < K; l++)
        {
            sum += A[row * K + l] * B[l * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matMul(const float *A, const float *B, float *C, const int M, const int K, const int N)
{
    nvtxRangePush("::Matrix Multiplication::");
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    nvtxRangePush("::Memory Allocation::");
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    nvtxRangePop();

    nvtxRangePush("::MemCpy:: Host -> Device");
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("::Kernel Execution::");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    nvtxRangePop();

    nvtxRangePush("::MemCpy::  Device -> Host");
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("::Memory deallocation::");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();
}

void init_matrix(float *mat, const int row, const int col)
{
    for (int i = 0; i < row * col; i++)
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    const int M = 1024, K = 1234, N = 1024;

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[N * N];

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    matMul(A, B, C, M, K, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}