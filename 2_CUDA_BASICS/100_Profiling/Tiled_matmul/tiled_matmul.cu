#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void tiledMatrixMult(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + ty;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        if (row < M && tile * TILE_SIZE + tx < K)
        {
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        }
    }
}