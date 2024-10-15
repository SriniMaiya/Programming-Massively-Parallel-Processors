#include <math.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_index < n)
    {
        C[data_index] = A[data_index] + B[data_index];
    }
}

void vecAdd(float *A, float *B, float *C, int n)
{
    float *A_d, *B_d, *C_d;
    int bytes = n * sizeof(float);

    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    int THREADBLOCK_SIZE = 256;
    int GRID_SIZE = (n + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    // int GRID_SIZE = (int)ceil((float)n / THREADBLOCK_SIZE);

    vecAddKernel<<<GRID_SIZE, THREADBLOCK_SIZE>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
}