#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cassert>
#include <algorithm>

__global__ void
conv1D(const int *input, const int *mask,
       int *result, const int n_inp, const int n_mask)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int kernel_radius = n_mask / 2;

    int kernel_start = thread_id - kernel_radius;
    int kernel_sum = 0;

    for (int j = 0; j < n_mask; j++)
    {
        if (((kernel_start + j) >= 0) && ((kernel_start + j) < n_inp))
        {
            kernel_sum += mask[j] * input[kernel_start + j];
        }
    }

    result[thread_id] = kernel_sum;
}

void verify_result(const int *input, const int *mask,
                   const int *result, const int n_inp, const int n_mask)
{
    int kernel_radius = n_mask / 2;
    int kernel_sum;
    int kernel_start;

    for (int i = 0; i < n_inp; i++)
    {
        kernel_sum = 0;
        kernel_start = i - kernel_radius;
        for (int j = 0; j < n_mask; j++)
        {
            if (((kernel_start + j) >= 0) && ((kernel_start + j) < n_inp))
            {
                kernel_sum += mask[j] * input[kernel_start + j];
            }
        }
        // printf("Sum CPU: %d, Sum GPU: %d\n", kernel_sum, result[i]);
        // fflush(stdout);
        assert(kernel_sum == result[i]);
    }
}

void init_vector(int *vec, const int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = rand() % 10;
    }
}

void init_mask(int *mask, const int mask_data[], const int n_mask)
{
    for (int m = 0; m < n_mask; m++)
    {
        mask[m] = mask_data[m];
    }
}

int main()
{
    int n_inp = 1 << 20;
    int bytes_ninp = n_inp * sizeof(int);

    int n_mask = 7;
    int bytes_nmask = n_mask * sizeof(int);

    srand(42);

    int *input;
    int *mask;
    int *result;

    const int mask_data[n_mask] = {1, 1, 1, 2, 1, 1, 1};

    cudaMallocManaged(&input, bytes_ninp);
    cudaMallocManaged(&mask, bytes_nmask);
    cudaMallocManaged(&result, bytes_ninp);

    init_vector(input, n_inp);
    init_mask(mask, mask_data, n_mask);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (n_inp + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1D<<<GRID_SIZE, BLOCK_SIZE>>>(input, mask, result, n_inp, n_mask);
    // cudaMemPrefetchAsync(result, bytes_ninp, cudaCpuDeviceId);

    cudaDeviceSynchronize();

    verify_result(input, mask, result, n_inp, n_mask);

    printf("COMPLETED SUCCESSFULLY");

    cudaFree(input);
    cudaFree(mask);
    cudaFree(result);

    return 0;
}
