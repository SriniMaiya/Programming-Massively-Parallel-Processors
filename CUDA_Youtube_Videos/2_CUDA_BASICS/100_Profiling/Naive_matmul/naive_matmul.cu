#include <iostream>
#include <cuda_runtime.h>

__global__ void naiveMatMul(const float *a, const float *b, float *c, const int M, const int K, const int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0;
        for (int l = 0; l < K; l++)
        {
            sum += a[row * K + l] + b[l * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    return 0;
}