#include <stdio.h>

__global__ void whoami(void)
{
    // The below represented 3Dim is usually ravelled in the GPU linearly. To find the id of the block,
    // SHAPE OF THE GRID: gridDim.x￼→, gridDim.y ↑, gridDim.z ↙
    // block_id: Read from bottom to top to understand better.
    int block_id =
        blockIdx.x +             /* 3. From the selected row, at which position(blockIdx.x) is the required block located?*/
        blockIdx.y * gridDim.x + /* 2. Given the length of the row (gridDim.x), which
                                 column(blockIdx.y) corresponds to the required row?*/

        blockIdx.z * gridDim.x * gridDim.y; /* 1. Given the slice with dimensions (gridDim.x ,gridDim.y)
                                                at which slice (blockIdx.z) is the given block located ?*/

    int block_offset =
        block_id *                            // Location of the current block
        blockDim.x * blockDim.y * blockDim.z; // Threads per block.

    // Which thread in the current block
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // ID of the thread in the whole grid.

    printf("ID: %04d | Block (%d, %d, %d) = %3d | Thread(%d, %d, %d) = %3d\n",
           id,
           blockIdx.x, blockIdx.y, blockIdx.z, block_offset,
           threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
};

int main()
{
    const int blockDim_x = 2, blockDim_y = 2, blockDim_z = 2;
    const int threadDim_x = 2, threadDim_y = 2, threadDim_z = 2;

    int blocks_per_grid = blockDim_x * blockDim_y * blockDim_z;
    int threads_per_block = threadDim_x * threadDim_y * threadDim_z;

    printf("Blocks/Grid: %d \n", blocks_per_grid);
    printf("Threads/Block: %d \n", threads_per_block);
    printf("Total threads: %d \n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(blockDim_x, blockDim_y, blockDim_z);
    dim3 threadsPerBlock(threadDim_x, threadDim_y, threadDim_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}
