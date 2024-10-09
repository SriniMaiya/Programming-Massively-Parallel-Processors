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

    // Compensating the halving of the blocks by adding the out-of-reach elements directly while loading the data into GPU.
    /*
        Ex:       blockDim : 2 (2 threads per block)
            size of vector : 8
            num. of blocks : ceil(8 / 2 * 2) = 2 ---> Reduce the number of blocks by 2.

            --> 2 blocks with 2 threads each.

            LOGIC :
            --> int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
            --> partial_sum[threadIdx.x] = vector[i] + vector[i + blockDim.x];


            :::::::::: THREAD BLOCK 1 ::::::::::
                1. blockIdx.x = 0, threadIdx.x = 0
                    ==> i = 0      (0 * (2 * 2) + 0)
                    partial_sum[0] = vector[0] + vector[0 + 2]; ---> Adding 1st and 3rd element directly while loading.

                2. blockIdx.x = 0, threadIdx.x = 1
                    ===> i = 1     (0 * (2 * 2) + 1)
                    partial_sum[1] = vector[1] + vector[1 + 2];  ---> Adding 2nd and 4th element directly while loading.


            :::::::::: THREAD BLOCK 2 ::::::::::
                3. blockIdx.x = 1, threadIdx.x = 0
                    ===> i = 4      (1 * (2 * 2) + 0 )
                    partial_sum[0] = vector[4] + vector[4 + 2];      ---> Adding 5th and 7th element directly while loading.

                4. blockIdx.x = 1, threadIdx.x = 1
                    ===> i = 5      (1 * (2 * 2) + 1)
                    partial_sum[1] = vector[5] + vector[5 + 2];      ---> Adding 6th and 8th element directly while loading.
    */
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // get the global element from vector (
    partial_sum[threadIdx.x] = vector[i] + vector[i + blockDim.x];
    // wait for all the threads to fetch and store the data in the SHMEM block.
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i = i >> 1)
    {
        if (threadIdx.x < i)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
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
    int num_blocks = (int)ceil(((float)N / threads_per_block) / 2);

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