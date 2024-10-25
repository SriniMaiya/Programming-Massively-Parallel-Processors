#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define RUNS 100
#define N 10000000
#define BLOCK_SIZE_1D 1024
#define Nx 1000
#define Ny 100
#define Nz 100
#define BLOCK_SIZE_3D_X 1024
#define BLOCK_SIZE_3D_Y 1024
#define BLOCK_SIZE_3D_Z 64

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
    // One multiply, one add, one store
    if (i < number)
    {
        c[i] = a[i] + b[i];
    } // One comparison, one addition, one store.
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // !!! 3 multiplies, 3 addition, 3 store operation !!!
    if (i < nx && j < ny && k < nz)
    {
        int index = i + j * nx + k * nx * ny;
        if (index < nx * ny * nz)
        {
            c[index] = a[index] + b[index];
        }
    } // !!! 4 comparisions, 5 multiplications, 3 additions, 1 store operation !!!
}

void init_vector(float *vector, int number)
{
    for (int i = 0; i < number; i++)
    {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_a, *h_b, *h_c_gpu1D, *h_c_gpu3D, *h_c_cpu;
    float *d_a, *d_b, *d_c;

    size_t size_vec = Width_N * sizeof(float);

    h_a = (float *)malloc(size_vec);
    h_b = (float *)malloc(size_vec);
    h_c_cpu = (float *)malloc(size_vec);
    h_c_gpu1D = (float *)malloc(size_vec);
    h_c_gpu3D = (float *)malloc(size_vec);

    srand(42);
    init_vector(h_a, Width_N);
    init_vector(h_b, Width_N);

    cudaMalloc(&d_a, size_vec);
    cudaMalloc(&d_b, size_vec);
    cudaMalloc(&d_c, size_vec);

    cudaMemcpy(d_a, h_a, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_vec, cudaMemcpyHostToDevice);

    int num_blocks_1d = (Width_N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    dim3 block_size3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (Nx + block_size3d.x - 1) / block_size3d.x,
        (Ny + block_size3d.y - 1) / block_size3d.y,
        (Nz + block_size3d.z - 1) / block_size3d.z);

    printf("Performing warm-up runs.... \n");

    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, Width_N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, Width_N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size3d>>>(d_a, d_b, d_c, Nx, Ny, Nz);
        cudaDeviceSynchronize();
    }

    printf("Performing CPU benchmark for %d runs.", RUNS);
    fflush(stdout);
    double cpu_time = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, Width_N);
        double end_time = get_time();
        cpu_time += (end_time - start_time);
        printf(".");
        fflush(stdout);
    }
    float avg_cpu_time = cpu_time * 1e3 / RUNS;

    printf("\nPerforming 1D - GPU benchmark for %d runs.", RUNS);
    fflush(stdout);
    double gpu_time_1d = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, Width_N);
        double end_time = get_time();
        gpu_time_1d += (end_time - start_time);
        printf(".");
        fflush(stdout);
    }
    cudaMemcpy(h_c_gpu1D, d_c, size_vec, cudaMemcpyDeviceToHost);
    double avg_gpu1d_time = gpu_time_1d * 1e3 / RUNS;

    printf("\nPerforming 3D - GPU benchmark for %d runs.", RUNS);
    fflush(stdout);
    double gpu_time_3d = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size3d>>>(d_a, d_b, d_c, Nx, Ny, Nz);
        double end_time = get_time();
        gpu_time_3d += (end_time - start_time);
        printf(".");
        fflush(stdout);
    }
    cudaMemcpy(h_c_gpu3D, d_c, size_vec, cudaMemcpyDeviceToHost);
    double avg_gpu3d_time = gpu_time_3d * 1e3 / RUNS;

    printf("\n\n:::::::Results:::::::\n-> Average time for vector addtion with 10 Million elements: \n");
    printf("%10s %.4f milliseconds\n", "CPU: ", avg_cpu_time);
    printf("%10s %.4f milliseconds\n", "GPU-1D: ", avg_gpu1d_time);
    printf("%10s %.4f milliseconds\n", "GPU-3D: ", avg_gpu3d_time);

    bool correct = true;
    for (int i = 0; i < Width_N; i++)
    {
        if ((fabs(h_c_cpu[i] - h_c_gpu1D[i]) > 1e-5) || (fabs(h_c_gpu1D[i] - h_c_gpu3D[i])) > 1e-5)
        {
            correct = false;
            break;
        }
    }
    printf("\n:::::::Speedup factors:::::::\n\n* %30s %.3f\n* %30s %.3f\n* %30s %.3f\n",
           "GPU-1D-op over CPU:",
           avg_cpu_time / avg_gpu1d_time,
           "GPU-3D-op over CPU:",
           avg_cpu_time / avg_gpu3d_time,
           "GPU-3D-op vs GPU-1D-op:",
           avg_gpu1d_time / avg_gpu3d_time);
    printf("\n\nAll types of addition produce the same result?\nh_c_cpu == h_c_gpu1D == h_c_gpu3D: %s\n", correct ? "true" : "false");
}
