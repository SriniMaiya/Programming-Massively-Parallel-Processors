#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <cassert>
#include <algorithm>

/* The logic for matrix multiplication assumes the following:
        - The matrix is square M == K == N
        - N // BLOCK_SIZE == 0

   Benchmarking of matrix multiplication using naive approach, tiled approach and tiled coalescing approach.
*/
using std::cout, std::endl;
using std::vector, std::generate;

const int N = 1 << 10;
const int SHMEM_SIZE = 16 * 16 * sizeof(int);

__global__ void naiveMatMul(const int *a, const int *b, int *c)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < N) && (col < N))
    {
        int temp_sum = 0;
        for (int k = 0; k < N; k++)
        {
            temp_sum += a[row * N + k] * b[k * row + col];
        }
        c[row * N + col] = temp_sum;
    }
}

__global__ void naiveCoalescedB(const int *a, const int *b_T, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < N) && (col < N))
    {
        int temp_sum = 0;
        for (int k = 0; k < N; k++)
        {
            temp_sum += a[row * N + k] * b[]
        }
    }
}