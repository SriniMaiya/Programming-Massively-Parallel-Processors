#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#define SIZE 256
#define SHMEM_SIZE SIZE * sizeof(int)

__global__ void sum_reduction(int *vector, int *vector_result)
{
    // shared memory block for partial sums
    __shared__ int partial_sum[SHMEM_SIZE];

    // global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // get the global element from vector (vector[tid]) to the corresponding local index of the SHMEM block.
    partial_sum[threadIdx.x] = vector[tid];
    // wait for all the threads to fetch and store the data in the SHMEM block.
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        int index = 2 * i * threadIdx.x;

        if (index < blockDim.x)
        {
            partial_sum[index] += partial_sum[index + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        vector_result[blockIdx.x] = partial_sum[0];
    }
}

void init_vector(int *vec, const int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = 1;
    }
}

int main()
{
    int N = 1 << 15;
    int bytes = N * sizeof(int);
    // host vetor and host vector result.
    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    h_v = (int *)malloc(bytes);
    h_v_r = (int *)malloc(bytes);

    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    init_vector(h_v, N);

    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    // thread block size <=> number of threads
    int threads_per_block = SIZE;

    // Grid size <=> number of threadblocks.
    int num_blocks = (int)ceil(N / threads_per_block);

    // Call kernel.
    sum_reduction<<<num_blocks, threads_per_block>>>(d_v, d_v_r);

    // reduce to scalar.
    sum_reduction<<<1, threads_per_block>>>(d_v_r, d_v_r);

    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Accumulated result: %d\n", h_v_r[0]);

    assert(h_v_r[0] == 1 << 15);
    printf("COMPILATION SUCCEEDED\n");

    return 0;
}