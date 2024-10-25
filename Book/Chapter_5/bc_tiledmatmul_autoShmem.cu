#include <stdio.h>

#define Height_M 2048
#define HeightWidth_MN 1024
#define Width_N 1536

// #define Height_M 4
// #define HeightWidth_MN 8
// #define Width_N 8

// #define M 256
// #define K 1024
// #define N 512

#define TILE_WIDTH 32
#define BLOCK_SIZE 32

/*
    The tiled version of matrix multiplication reduces the global memory accesses by a factor of
    TILE_WIDTH.
    Ex: With 16 X 16 tiles, the global memory access time is reduced by a factor of 16.

*/
__global__ void matrixMultKernel(float *M, float *N, float *P)
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

    // 🔲 Runs for all blocks in the grid.
    int num_phases = (HeightWidth_MN + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ph++)
    {
        // Per thread, load values into shared memory
        if ((Row < Height_M) && (ph * TILE_WIDTH + tx) < HeightWidth_MN)
        {
            Mds[ty][tx] = M[Row * HeightWidth_MN + (ph * TILE_WIDTH + tx)];
        }
        else
            Mds[ty][tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < HeightWidth_MN && Col < Width_N)
        {
            Nds[ty][tx] = N[ph * TILE_WIDTH * Width_N + ty * Width_N + Col];
        }
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        // Runs for all the threads in the current block. 🧵
        for (int k = 0; k < TILE_WIDTH && k < HeightWidth_MN; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }
    // Once the Pvalue is calculated @P[Row, Col] substitute the calculated Pvalue.
    if (Row < Height_M && Col < Width_N)
        P[Row * Width_N + Col] = Pvalue;
}

void matmult_cpu(const float *A, const float *B, float *C, const int m, const int n, const int k)
{
    // const int a_size = sizeof(A) / sizeof(A[0]);
    // printf("Size of a %zu \tnum_el: %d\n", sizeof(A), a_size);
    // fflush(stdout);
    // int b_size = sizeof(B) / sizeof(B[0]);
    // assert(a_size == m * k);
    // assert(b_size == k * n);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
                sum += (A[i * k + l] * B[l * n + j]);
                // i * n -> Which row of A
                // k     -> Which element of A in the row.
                // Add both to get unraveled position of the matrix.
            }
            C[i * n + j] = sum;
        }
    }
}

void print_matrix(const char *name, const float *mat, const int r, const int c)
{
    printf("::::::%s::::::: \n", name);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%*.3f ", 7, mat[i * c + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        // mat[i] = (float)rand() / RAND_MAX;
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_a = Height_M * HeightWidth_MN * sizeof(float);
    int size_b = HeightWidth_MN * Width_N * sizeof(float);
    int size_c = Height_M * Width_N * sizeof(float);

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_cpu = (float *)malloc(size_c);
    h_c_gpu = (float *)malloc(size_c);

    srand(42);
    init_matrix(h_a, Height_M, HeightWidth_MN);
    init_matrix(h_b, HeightWidth_MN, Width_N);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // Number for threads per block: 16*16= 256
    dim3 gridDim((Width_N + BLOCK_SIZE - 1) / BLOCK_SIZE, (Height_M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c);
    matmult_cpu(h_a, h_b, h_c_cpu, Height_M, Width_N, HeightWidth_MN);
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < Height_M * Width_N; i++)
    {
        // printf("Sizeof CPU: %zu GPU: %zu", sizeof(h_c_cpu), sizeof(h_c_gpu));
        // printf("CPU: %f GPU: %f\n", h_c_cpu[i], h_c_gpu[i]);
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-3)
        {
            correct = false;
            break;
        }
    }
    if (Height_M < 15)
    {
        print_matrix((char *)"M", h_a, Height_M, HeightWidth_MN);
        print_matrix((char *)"N", h_b, HeightWidth_MN, Width_N);
        print_matrix((char *)"P_CPU", h_c_cpu, Height_M, Width_N);
        print_matrix((char *)"P_GPU", h_c_gpu, Height_M, Width_N);
    }

    // Printing

    // print_matrix((char *)"h_a", h_a, Height_M, HeightWidth_MN);
    // print_matrix((char *)"h_b", h_b, HeightWidth_MN, Width_N);
    // print_matrix((char *)"h_c_cpu", h_c_cpu, Height_M, Width_N);
    // print_matrix((char *)"h_c_gpu", h_c_gpu, Height_M, Width_N);

    printf("The CPU and GPU results are %s\n", correct ? "matching" : "not matching");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}