// Datatransfer from DRAM is slow!
// A method of caching on GPU would enhance performance, as there is always something
// in cache that has to be processed.
// ==> Fewer memory related stalls ==> More computation/unit time.

//::SOLUTION::
//* Shared Memory (User Managed L1 Cache.)
//  ---> Private per threadblock.

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <functional>
#include <algorithm>

#include <cuda_runtime.h>

using std::cout;
using std::generate;
using std::vector;

// Matrix and shared memory tile size
const int N = 1 << 10;
const int SHMEM_SIZE = 16 * 16 * sizeof(int);

__global__ void matrixMul(const int *a, const int *b, int *c)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Static shared memory tiles.
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    int tmp = 0;

    for (int i = 0; i < N; i += blockDim.x)
    {
        // load elements of the tile.
        /*
         s_a : Tile of matrix a. a is accessed row-wise.
         s_b : Tile of matrix b. b is accessed column-wise.
        */
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++)
        {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        c[row * N + col] = tmp;
    }
}

// Check result on the CPU
void verify_result(const vector<int> &a, const vector<int> &b, const vector<int> &c)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            // cout << "Sum: " << sum << " c[i * N + J]: " << c[i * N + j] << std::endl << std::flush;
            assert(c[i * N + j] == sum);
        }
    }
}

int main()
{
    size_t elements = N * N;
    size_t bytes = elements * sizeof(int);

    vector<int> h_a(elements);
    vector<int> h_b(elements);
    vector<int> h_c(elements);

    generate(h_a.begin(), h_a.end(), []()
             { return rand() % 100; });

    generate(h_b.begin(), h_b.end(), []()
             { return rand() % 100; });

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 32;
    int NUM_BLOCKS = N / THREADS_PER_BLOCK;

    dim3 num_threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 num_blocks(NUM_BLOCKS, NUM_BLOCKS);

    matrixMul<<<num_blocks, num_threads>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY:::NO ASSERTION ERRORS. " << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "CUDA ERROR: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    return 0;
}