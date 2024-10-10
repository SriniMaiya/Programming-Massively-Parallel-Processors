#include <cooperative_groups.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <iostream>

namespace cg = cooperative_groups;

__device__ int thread_sum(int *input, int n)
{
    int sum = 0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // i += gridDim.x * blockDim.x ==> Signifies, if the data does not fit on the
    for (int i = thread_id; i < n / 4; i += gridDim.x * blockDim.x)
    {
        int4 *int4Grouped_input = (int4 *)input;
        int4 indexed_int4 = int4Grouped_input[i];

        sum += indexed_int4.x + indexed_int4.y + indexed_int4.z + indexed_int4.w;
    }

    return sum;
}

__device__ int reduce_block(cg::thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();

    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync();

        if (lane < i)
        {
            val += temp[lane + i];
        }

        g.sync();
    }
    return val;
}

__global__ void sum_reduction(unsigned long long int *sum, int *input, int n)
{
    int partial_sum = thread_sum(input, n);

    extern __shared__ int temp[];
    auto g = cg::this_thread_block();

    int block_sum = reduce_block(g, temp, partial_sum);

    // printf("threadIdx: %d\tpartial_sum: %d\tblock_sum: %d\n", threadIdx.x, partial_sum, block_sum);

    if (g.thread_rank() == 0)
    {
        atomicAdd(sum, block_sum);
    }
}

void initializa_vector(int *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = i;
        // vec[i] = rand() % 10;
    }
}

int main()
{
    int N = 1 << 20;
    int bytes = N * sizeof(int);

    int *vec;
    unsigned long long int *sum;

    cudaMallocManaged(&vec, bytes);
    cudaMallocManaged(&sum, sizeof(long int));

    initializa_vector(vec, N);

    int TB_SIZE = 256;

    int GRID_SIZE = (N + TB_SIZE - 1) / TB_SIZE;

    sum_reduction<<<GRID_SIZE, TB_SIZE, TB_SIZE * sizeof(int)>>>(sum, vec, N);

    cudaDeviceSynchronize();
    printf("Accumulated sum: %lld \n", *sum);

    // assert(*sum == N);

    return 0;
}