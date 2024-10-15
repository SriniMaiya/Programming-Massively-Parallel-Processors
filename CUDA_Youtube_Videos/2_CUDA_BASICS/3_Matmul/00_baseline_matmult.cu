#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>
#include <string>

#define RUNS 5
#define M 256
#define K 1024
#define N 512

#define BLOCK_SIZE 32

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

__global__ void matmul_gpu(const float *A, const float *B, float *C, const int m, const int n, const int k)
{
    // Single Instruction Multiple Thread Model (SIMT)
    // Here we are concerned with calculation of the element of only a single thread.
    // The GPU scales/replicates the same instruction on all the assigned thread for parallel computation.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void print_matrix(char *name, float *mat, int r, int c)
{
    printf("::::::%s::::::: \n", name);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%.3f ", mat[i * c + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_cpu = (float *)malloc(size_c);
    h_c_gpu = (float *)malloc(size_c);

    srand(42);
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // Number for threads per block: 16*16= 256
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Performing warm-up runs...\n");
    {
        for (int i = 0; i < 3; i++)
        {
            matmult_cpu(h_a, h_b, h_c_cpu, M, N, K);
            matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
            cudaDeviceSynchronize();
        }
    }
    printf("Benchmarking on CPU --> %d runs..... \n", RUNS);
    double cpu_time = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        matmult_cpu(h_a, h_b, h_c_cpu, M, N, K);
        double end_time = get_time();
        cpu_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_time / RUNS;

    printf("Benchmarking on GPU --> %d runs..... \n", RUNS);
    double gpu_time = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        matmul_gpu<<<blockDim, gridDim>>>(d_a, d_b, d_c, M, N, K);
        double end_time = get_time();
        gpu_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_time / RUNS;
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1e3);
    printf("GPU average time: %lf milliseconds\n", gpu_avg_time * 1e3);
    printf("Speedup of GPU over CPU: %f\n", cpu_avg_time / gpu_avg_time); // For addition of vectors of 10 Million,
                                                                          // a speed-up of 1e4 is noticed.

    // Verify is both CPU and GPU operations yield the same result.
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        // printf("Sizeof CPU: %zu GPU: %zu", sizeof(h_c_cpu), sizeof(h_c_gpu));
        // printf("CPU: %f GPU: %f\n", h_c_cpu[i], h_c_gpu[i]);
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-3)
        {
            correct = false;
            break;
        }
    }

    // Printing

    // print_matrix((char *)"h_a", h_a, M, K);
    // print_matrix((char *)"h_b", h_b, K, N);
    // print_matrix((char *)"h_c_cpu", h_c_cpu, M, N);
    // print_matrix((char *)"h_c_gpu", h_c_gpu, M, N);

    printf("The CPU and GPU results are %s\n", correct ? "matching" : "not matching");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}