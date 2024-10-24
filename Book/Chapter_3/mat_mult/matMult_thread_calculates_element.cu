#include <stdio.h>
#include <assert.h>
// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

void print_mat(float *, int, int);

__global__ void matMultKernel(float *M, float *N, float *P, int height_m, int k, int width_n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < height_m) && (col < width_n))
    {
        float sum = 0;
        for (int K = 0; K < k; K++)
        {
            sum += M[row * k + K] * N[K * width_n + col];
        }
        P[row * width_n + col] = sum;
    }
}

void init_mat(float *mat, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

void check_results(float *M, float *N, float *P, int height_m, int k, int width_n)
{
    // printf("::: M :::\n");
    // print_mat(M, height_m, k);
    // printf("::: N :::\n");
    // print_mat(N, k, width_n);
    // printf("::: P :::\n");
    // print_mat(P, height_m, width_n);
    // printf("\n");
    for (int r_m = 0; r_m < height_m; r_m++)
    {
        for (int c_n = 0; c_n < width_n; c_n++)
        {
            float sum = 0;
            for (int K = 0; K < k; K++)
            {
                sum += M[r_m * k + K] * N[K * width_n + c_n];
            }
            // printf("%f == %f", sum, P[r_m * width_n + c_n]);
            // printf("%f", sum);
            // if (c_n == (width_n - 1))
            // {
            //     printf("\n");
            // }
            // else
            // {
            //     printf(" ");
            // }
            assert(abs(sum - P[r_m * width_n + c_n]) < 1e-4);
        }
    }
}

void print_mat(float *mat, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            printf("%f ", mat[r * cols + c]);
        }
        printf("\n");
    }
}

int main()
{
    float *M_d, *N_d, *P_d, *M_h, *N_h, *P_h;
    int height_m = 1024; // height of m
    int k = 1024;        // width of m; height of n.
    int width_n = 512;   // width of n

    size_t bytes_M = sizeof(float) * height_m * k;
    size_t bytes_N = sizeof(float) * k * width_n;
    size_t bytes_P = sizeof(float) * height_m * width_n;

    M_h = (float *)malloc(bytes_M);
    N_h = (float *)malloc(bytes_N);
    P_h = (float *)malloc(bytes_P);

    init_mat(M_h, height_m * k);
    init_mat(N_h, k * width_n);

    cudaMalloc(&M_d, bytes_M);
    cudaMalloc(&N_d, bytes_N);
    cudaMalloc(&P_d, bytes_P);

    cudaMemcpy(M_d, M_h, bytes_M, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, bytes_N, cudaMemcpyHostToDevice);

    int THREADS = 16;
    int GRID_SIZE_X = (width_n + THREADS - 1) / THREADS;
    int GRID_SIZE_Y = (height_m + THREADS - 1) / THREADS;

    dim3 BLOCK_SIZE(THREADS, THREADS);
    dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);

    matMultKernel<<<GRID_SIZE, BLOCK_SIZE>>>(M_d, N_d, P_d, height_m, k, width_n);

    cudaMemcpy(P_h, P_d, bytes_P, cudaMemcpyDeviceToHost);

    check_results(M_h, N_h, P_h, height_m, k, width_n);

    free(M_h);
    free(N_h);
    free(P_h);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    printf("COMPLETED SUCCESSFULLY\n");
    return 0;
}