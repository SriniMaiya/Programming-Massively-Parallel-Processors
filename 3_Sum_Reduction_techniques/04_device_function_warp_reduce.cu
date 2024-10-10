#include <iostream>
#include <cassert>
#include <cmath>

#define SIZE 256
#define SHMEM_SIZE SIZE * sizeof(int)

__device__ void wrapReduce(int *shmem_pointer, int threadID)
{
    for (int i = warpSize; i > 0; i = i >> 1)
    {
        shmem_pointer[threadID] += shmem_pointer[threadID + i];
    }
}

__global__ void sum_reduction(volatile int *vector, int *vector_result)
{
    // shared memory block for partial sums
    __shared__ int partial_sum[SHMEM_SIZE];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // get the global element from vector (
    partial_sum[threadIdx.x] = vector[i] + vector[i + blockDim.x];
    // wait for all the threads to fetch and store the data in the SHMEM block.
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i = i >> 1)
    {
        if (threadIdx.x < i)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x < warpSize)
    {
        wrapReduce(partial_sum, threadIdx.x);
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
    int N = 1 << 16;
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
    int num_blocks = (int)ceil((float)((float)N / threads_per_block) / 2);

    // Call kernel.
    sum_reduction<<<num_blocks, threads_per_block>>>(d_v, d_v_r);

    // reduce to scalar.
    sum_reduction<<<1, threads_per_block>>>(d_v_r, d_v_r);

    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Accumulated result: %d\n", h_v_r[0]);

    assert(h_v_r[0] == 1 << 16);
    printf("COMPILATION SUCCEEDED\n");

    return 0;
}