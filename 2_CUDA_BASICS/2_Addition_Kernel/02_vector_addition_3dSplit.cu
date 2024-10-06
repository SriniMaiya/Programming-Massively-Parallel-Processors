#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 16
#define BLOCK_SIZE_3D_Z 8

void vector_add_cpu(float *a, float *b, float *c, int number)
{
    for (int i = 0; i < number; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int index = i + j * nx + k * nx * ny;
        if (index < nx * ny * nz)
        {
            c[index] = a[index] + b[index];
        }
    }
}