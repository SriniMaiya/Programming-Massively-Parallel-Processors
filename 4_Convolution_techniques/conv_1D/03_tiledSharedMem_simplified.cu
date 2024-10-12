#include <stdio.h>
#include <assert.h>

#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];

__global__ void conv1D(const int *array, int *result, const int n)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int s_array[];

    s_array[threadIdx.x] = array[thread_id];
    __syncthreads();

    int sum = 0;

    for (int j = 0; j < MASK_LENGTH; j++)
    {
        if ((threadIdx.x + j) >= blockDim.x)
        {
            sum += array[thread_id + j] * mask[j];
        }
        else
        {
            sum += s_array[threadIdx.x + j] * mask[j];
        }
    }

    result[thread_id] = sum;
}

void verify_results(const int *array, const int *mask, const int *result, const int n)
{
    int sum;
    for (int i = 0; i < n; i++)
    {
        sum = 0;
        for (int j = 0; j < MASK_LENGTH; j++)
        {
            // printf("array[i+j]: %d, mask[j]: %d\n", array[i + j], mask[j]);
            sum += array[i + j] * mask[j];
        }
        // printf("GPU: %d, CPU: %d\n", result[i], sum);
        fflush(stdout);
        assert(sum == result[i]);
    }
}

int main()
{
    int n = 1 << 20;
    int mask_r = MASK_LENGTH / 2;
    int mask_d = 2 * mask_r;
    int n_pad = n + mask_d;

    int bytes_pad = sizeof(int) * n_pad;
    int bytes_arr = sizeof(int) * n;
    int bytes_mask = sizeof(int) * MASK_LENGTH;

    int *h_array = new int[n_pad];
    for (int i = 0; i < n_pad; i++)
    {
        if ((i < mask_r) || (i >= (n + mask_r)))
            h_array[i] = 0;
        else
            h_array[i] = rand() % 100;
    }

    int h_mask[MASK_LENGTH] = {1, 1, 1, 2, 1, 1, 1};
    int *h_result = new int[n];

    int *d_a, *d_r;
    cudaMalloc(&d_a, bytes_pad);
    cudaMalloc(&d_r, bytes_arr);

    cudaMemcpy(d_a, h_array, bytes_pad, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, bytes_mask);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int SHMEM = (BLOCK_SIZE + mask_d) * sizeof(int);

    conv1D<<<GRID_SIZE, BLOCK_SIZE, SHMEM>>>(d_a, d_r, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_r, bytes_arr, cudaMemcpyDeviceToHost);

    verify_results(h_array, h_mask, h_result, n);

    printf("COMPLETED SUCCESSFULLY");

    delete[] h_array;
    h_array = NULL;
    delete[] h_result;
    h_result = NULL;

    cudaFree(d_a);
    cudaFree(d_r);

    return 0;
}