#include <stdio.h>
#include <cstdlib>
#include <cassert>

#define MASK_LENGTH 7

// Thread debugging in CUDA:
// nvcc -G -O0 02_shared_mem.cu
// Set breakpoint in the __global__ function (kernel) --> Debug.

__constant__ int mask[MASK_LENGTH];

__global__ void conv1D(const int *array, int *result, const int n)
{
    // ID of the current thread.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Dynamic allocation of shared memory.
    extern __shared__ int s_array[];

    // Radius and diameter of the mask.
    int mask_radius = MASK_LENGTH / 2;
    int diameter = mask_radius * 2;

    // Size of the shared memory (To include the edge case).
    int n_padded = blockDim.x + diameter;

    // Which element should the thread point next. --> In shared memory (s_array)
    int offset = threadIdx.x + blockDim.x;
    // The offset globally.  ---> Which element in the global memory (array)
    int global_offset = blockIdx.x * blockDim.x + offset;

    // Load the lower elemnts first starting at the edge case (halo)
    s_array[threadIdx.x] = array[global_tid];

    if (offset < n_padded)
    {
        s_array[offset] = array[global_offset];
    }

    __syncthreads();

    int sum = 0;

    for (int j = 0; j < MASK_LENGTH; j++)
    {
        sum += s_array[threadIdx.x + j] * mask[j];
    }

    result[global_tid] = sum;
}

void verify_result(const int *array, const int *mask, const int *result, const int n)
{
    int sum;
    for (int i = 0; i < n; i++)
    {
        sum = 0;
        for (int j = 0; j < MASK_LENGTH; j++)
        {
            sum += mask[j] * array[i + j];
        }

        assert(sum == result[i]);
    }
}

int main()
{
    int n = 1 << 20;

    int radius = MASK_LENGTH / 2;
    int diameter = radius * 2;

    int n_pad = n + diameter;

    size_t bytes_arr = n * sizeof(int);
    size_t bytes_arrPad = n_pad * sizeof(int);
    size_t bytes_mask = MASK_LENGTH * sizeof(int);

    int *h_array = new int[n_pad];
    for (int i = 0; i < n_pad; i++)
    {
        if ((i < radius) || (i >= (n + radius)))
        {
            h_array[i] = 0;
        }
        else
        {
            h_array[i] = rand() % 100;
        }
    }

    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++)
    {
        h_mask[i] = rand() % 10;
    }

    int *h_result = new int[n];

    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_arrPad);
    cudaMalloc(&d_result, bytes_arrPad);

    cudaMemcpy(d_array, h_array, bytes_arrPad, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(mask, h_mask, bytes_mask, 0, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t SHMEM = (BLOCK_SIZE + diameter) * sizeof(int);

    conv1D<<<GRID_SIZE, BLOCK_SIZE, SHMEM>>>(d_array, d_result, n);

    cudaMemcpy(h_result, d_result, bytes_arr, cudaMemcpyDeviceToHost);

    verify_result(h_array, h_mask, h_result, n);

    printf("COMPLETED SUCCESSFULLY\n");

    delete[] h_array;
    delete[] h_mask;
    delete[] h_result;

    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}