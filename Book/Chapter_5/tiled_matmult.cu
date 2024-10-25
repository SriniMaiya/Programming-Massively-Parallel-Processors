#include <stdio.h>
#define TILE_WIDTH 16
/*
    The tiled version of matrix multiplication reduces the global memory accesses by a factor of
    TILE_WIDTH.
    Ex: With 16 X 16 tiles, the global memory access time is reduced by a factor of 16.

*/
__global__ void matrixMultKernel(float *M, float *N, float *P, int Width)
{
  // Each thread calculates one element of the P matrix.
  // The kernel is executed per block.

  /*
   Mds : TILE_WIDTH**2 number of M elements
   Nds : TILE_WIDTH**2 number of N elements
  */
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // For each threads; bx, by, tx, ty -> Reside in registers
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Row and Column values in the block.
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // For each thread a Pvalue is defined.
  float Pvalue = 0;
  /*
   There are Width / TILE_WIDTH number of phases, which are iterated through the for loop.
   Each phase uses one tile of M and one tile of N elements.
   ph : indicates the number of phases that have already been done for the dot product.
   For every thread in the shared memory block, load the M & N values into Mds and Nds.
  */
  // 🔲 Runs for all blocks in the grid.
  for (int ph = 0; ph < Width / TILE_WIDTH; ph++)
  {
    // Per thread, load values into shared memory
    /*
      Getting the global index of M.
      Rows: Row
      Cols: ph * TILE_WIDTH + tx
      ::::::::::::::::::::::::::
      Row                   ->  Which row.
      ph * TILE_WIDTH + tx  -> (Num phase) * (width of the tile) + Local col index in the block.
    */
    Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
    /*
      Getting the global index of N.
      Rows: (ph * TILE_WIDTH + ty)
      Cols: Col
      ::::::::::::::::::::::::::
      ph * TILE_WIDTH + ty -> (Num phase) * (width of the tile) + Local row index in the block.
      Col                  -> Which globla column.
    */
    Nds[ty][tx] = N[ph * TILE_WIDTH * Width + ty * Width + Col];
    // Barrier to let all the threads in the block to load elements into shared memory Mds, Nds.
    // Read-After-Write dependence. The threads have to wait for the data to be written before reading it.
    // In this case: true dependence. The data has to be written before the data is read.
    __syncthreads();

    // Runs for all the threads in the current block. 🧵
    for (int k = 0; k < TILE_WIDTH; k++)
    {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    // Barrier to let all threads to calculate and append to the Pvalue for the block.
    // Write-After-Read depencence. The data has to be read before the data has to be written/overwritten.
    // In this case: false dependence. Because writing thread does not need any data from reading thread.
    // Synchronization is needed because the same memory location is being reused.
    __syncthreads();
  }
  // Once the Pvalue is calculated @P[Row, Col] substitute the calculated Pvalue.
  P[Row * Width + Col] = Pvalue;
}

/*
The loop nest from line 35 to line 73 illustrates a technique called strip-mining, which takes a long-running loop and break it into phases. Each phase involves an inner loop that executes a few consecutive iterations of the original loop. The original loop becomes an outer loop whose role is to iteratively invoke the inner loop so that all the iterations of the original loop are executed in their
original order. By adding barrier synchronizations before and after the inner loop, we force all threads in the same block to focus their work on the same section of input data during each phase.
*/