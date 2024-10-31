#include <stdio.h>
#include <string>
#include <sstream>

using std::string;

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matMultKernel(float *M, float *N, float *P, int Width)
{
    size_t size_addr = sizeof(static_cast<const void *>(&M[0]));

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR] = {0.0f};

    int num_phases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < num_phases; ph++)
    {
        if (row < Width && (ph * TILE_WIDTH + tx) < Width)
        {
            Mds[ty][tx] = M[row * Width + (ph * TILE_WIDTH + tx)];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; c++)
        {
            int col = colStart + c * TILE_WIDTH;
            if ((ph * TILE_WIDTH + ty) < Width && col < Width)
            {
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + col];
            }
            else
            {
                Nds[ty][tx] = 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; k++)
            {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int col = colStart + c * TILE_WIDTH;
        if (col < Width)
        {
            P[row * Width + col] = Pvalue[c];
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

void save_matLayout(const float *mat, const int r, const int c, const char *name)
{
    string path = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/Book/Chapter_6/";
    path.append(name);

    FILE *f = fopen(path.c_str(), "w");
    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            fprintf(f, "%p ", (void *)&mat[row * r + col]);
        }
        fprintf(f, "%s", "\n");
    }
    fclose(f);
}

void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        // mat[i] = (float)rand() / RAND_MAX;
        mat[i] = (float)rand() / RAND_MAX;
    }
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

int main()
{
    int Width = 5;
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_a = Width * Width * sizeof(float);
    int size_b = Width * Width * sizeof(float);
    int size_c = Width * Width * sizeof(float);

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_cpu = (float *)malloc(size_c);
    h_c_gpu = (float *)malloc(size_c);

    srand((unsigned int)4815163642);
    init_matrix(h_a, Width, Width);
    init_matrix(h_b, Width, Width);

    save_matLayout(h_a, Width, Width, "Matrix_a.txt");
    save_matLayout(h_b, Width, Width, "Matrix_b.txt");

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Number for threads per block: 16*16= 256
    dim3 gridDim((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    matMultKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, Width);
    matmult_cpu(h_a, h_b, h_c_cpu, Width, Width, Width);
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < Width * Width; i++)
    {
        // printf("Sizeof CPU: %zu GPU: %zu", sizeof(h_c_cpu), sizeof(h_c_gpu));
        // printf("CPU: %f GPU: %f\n", h_c_cpu[i], h_c_gpu[i]);
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-3)
        {
            correct = false;
            break;
        }
    }
    if (Width <= 10)
    {
        print_matrix((char *)"M", h_a, Width, Width);
        print_matrix((char *)"N", h_b, Width, Width);
        print_matrix((char *)"P_CPU", h_c_cpu, Width, Width);
        print_matrix((char *)"P_GPU", h_c_gpu, Width, Width);
    }
    // Printing

    // print_matrix((char *)"h_a", h_a, Width, Width);
    // print_matrix((char *)"h_b", h_b, Width, Width);
    // print_matrix((char *)"h_c_cpu", h_c_cpu, Width, Width);
    // print_matrix((char *)"h_c_gpu", h_c_gpu, Width, Width);

    printf("The CPU and GPU results are %s\n", correct ? "matching" : "not matching");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}