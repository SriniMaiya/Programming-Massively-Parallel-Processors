#include <math.h>
#include <cassert>
#include <stdio.h>

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

void init_vector(float *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = (float)(rand() % RAND_MAX) / RAND_MAX;
    }
}

void check_results(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        assert(C[i] = A[i] + B[i]);
    }
    printf("COMPLETED SUCCESSFULLY\n");
}

int main()
{
    float *A, *B, *C;
    int n = 1 < 10;
    int bytes = n * sizeof(float);

    A = (float *)malloc(bytes);
    B = (float *)malloc(bytes);
    C = (float *)malloc(bytes);

    init_vector(A, n);
    init_vector(B, n);

    vecAdd(A, B, C, n);

    check_results(A, B, C, n);
}