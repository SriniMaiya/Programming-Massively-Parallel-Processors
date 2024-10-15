#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

__global__ void vectorAddUM(const int *a, const int *b, int *c, const size_t N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < 10)
    {
        printf("gridDim.x: %d, blockDim.x: %d\n", gridDim.x, blockDim.x);
    }

    if (thread_id < N)
    {
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}

void initVector(int *vector, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        vector[i] = rand() % (100 - 0 + 1) + 0;
    }
}

void verifyAnswer(const int *a, const int *b, const int *c, const int N)
{
    for (int i = 0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

double getTime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{

    const int N = 1 << 26;
    const int bytes = N * sizeof(int);
    int *a, *b, *c;

    // The memory allocation is totally upto cuda.
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    double start = getTime();
    initVector(a, N);
    initVector(b, N);

    int THREADS_PER_BLOCK = 256;

    // int GRID_SIZE = (int)ceil((float)N / THREADS_PER_BLOCK);
    int GRID_SIZE = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("GRID_SIZE: %d, NUM_THREADS: %d\n\n", GRID_SIZE, THREADS_PER_BLOCK);

    vectorAddUM<<<GRID_SIZE, THREADS_PER_BLOCK>>>(a, b, c, N);

    cudaDeviceSynchronize();

    verifyAnswer(a, b, c, N);

    double end = getTime();
    printf("Initialization to verification of Vector(%d elements) took %lf miliseconds.\n", N, (end - start) * 1e3);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
