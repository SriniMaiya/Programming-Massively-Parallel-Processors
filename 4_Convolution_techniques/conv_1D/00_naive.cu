#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

__global__ void conv1D(const int *input, const int *mask, int *result,
                       const int n_inp, const int n_mask)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    int mask_radius = n_mask / 2;
    int mask_start = global_tid - mask_radius;

    for (int j = 0; j < n_mask; j++)
    {
        if (((mask_start + j) >= 0) && ((mask_start + j) < n_inp))
        {
            sum += mask[j] * input[mask_start + j];
        }
    }

    result[global_tid] = sum;
}

void verify_results(const int *input, const int *mask, const int *result,
                    const int n_inp, const int n_mask)
{
    for (int i = 0; i < n_inp; i++)
    {
        int sum = 0;
        int mask_radius = n_mask / 2;
        int mask_start = i - mask_radius;

        for (int j = 0; j < n_mask; j++)
        {
            if (((mask_start + j) >= 0) && ((mask_start + j) < n_inp))
            {
                sum += mask[j] * input[mask_start + j];
            }
        }
        // printf("CPU: %d, GPU: %d \n", sum, result[i]);
        assert(sum == result[i]);
    }
}

int main()
{
    int n_inp = 2 << 20;
    int n_mask = 7;
    int bytes_inp = n_inp * sizeof(int);
    int bytes_mask = n_mask * sizeof(int);

    // Host data.
    std::vector<int> h_inp(n_inp), h_mask(n_mask), h_res(n_inp);

    std::generate(h_inp.begin(), h_inp.end(), []()
                  { return rand() % 10; });
    h_mask = {1, 1, 1, 2, 1, 1, 1};

    // Device data.
    int *d_inp, *d_mask, *d_res;

    cudaMalloc(&d_inp, bytes_inp);
    cudaMalloc(&d_res, bytes_inp);
    cudaMalloc(&d_mask, bytes_mask);

    // Copy host -> device.
    cudaMemcpy(d_inp, h_inp.data(), bytes_inp, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_mask, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (n_inp + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Kernel run.
    conv1D<<<GRID_SIZE, BLOCK_SIZE>>>(d_inp, d_mask, d_res, n_inp, n_mask);

    cudaDeviceSynchronize();

    // Copy device -> host.
    cudaMemcpy(h_res.data(), d_res, bytes_inp, cudaMemcpyDeviceToHost);

    verify_results(h_inp.data(), h_mask.data(), h_res.data(), n_inp, n_mask);

    cudaFree(d_inp);
    cudaFree(d_mask);
    cudaFree(d_res);

    return 0;
}