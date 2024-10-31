#include <stdio.h>

#define TILE_SIZE 32

__global__ void matMultKernel(float *M, float *N, float *P, int Width)
{
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    int num_phases = (Width + TILE_SIZE - 1) / TILE_SIZE;
    float Pvalue = 0.0f;

    for (int ph = 0; ph < num_phases; ph++)
    {
        if (row < Width && (bx * TILE_SIZE + tx) < Width)
        {
            Mds[ty][tx] = M[row * Width + (ph * TILE_SIZE + tx)];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }

        if ((ph * TILE_SIZE + ty) < Width && col < Width)
        {
            Nds[ty][tx] = N[(ph * TILE_SIZE + ty) * Width + col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if (row < Width && col < Width)
    {
        P[row * Width + col] = Pvalue;
    }
}