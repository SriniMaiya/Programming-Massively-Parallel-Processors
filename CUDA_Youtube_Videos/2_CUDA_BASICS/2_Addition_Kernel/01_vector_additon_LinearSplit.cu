#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define RUNS 20
#define N 10000000 // Vector size: 10 million
#define THREADS_PER_BLOCK 256

// cpu vector addtion

void vector_add_cpu(float *a, float *b, float *c, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addtion:
__global__ void vector_add_gpu(float *a, float *b, float *c, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

// Vector initialization
void init_vector(float *vector, size_t n)
{
    for (int i = 0; i < n; i++)
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
    /*
        Vector addition. Vec(c) = Vec(a) + Vec(b)

        ON CPU: float *host_a -> Vec(a)
                float *host_b -> Vec(b)
                float *host_c_cpu -> Vec(c)
                float *host_c_gpu -> Vec(a)

        ON GPU: float *device_a -> copy of host_a on GPU
                float *device_b -> copy of host_b on GPU
                float *device_c -> device_a + device_b on GPU
    */
    float *host_a, *host_b, *host_c_cpu, *host_c_gpu;
    float *device_a, *device_b, *device_c;
    size_t size_vec = N * sizeof(float); // Size of the whole vector.

    // Cast allocated memory to float*
    host_a = (float *)malloc(size_vec);
    host_b = (float *)malloc(size_vec);
    host_c_cpu = (float *)malloc(size_vec);
    host_c_gpu = (float *)malloc(size_vec);

    srand(time(NULL));

    // Initialize vectors a, b on cpu.
    init_vector(host_a, N);
    init_vector(host_b, N);

    // Allocate CUDA memory.
    printf("Allocating CUDA memory....\n");
    cudaMalloc(&device_a, size_vec);
    cudaMalloc(&device_b, size_vec);
    cudaMalloc(&device_c, size_vec);

    // Copy host_a, host_b to CUDA
    cudaMemcpy(device_a, host_a, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size_vec, cudaMemcpyHostToDevice);

    // Always keeps 1 additional block during the  (N/THREADS_PER_BLOCK) division with remainder.
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Performing warm-up runs.....\n");
    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(host_a, host_b, host_c_cpu, N);
        vector_add_gpu<<<num_blocks, THREADS_PER_BLOCK>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking on CPU --> %d runs..... \n", RUNS);
    double cpu_time = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        vector_add_cpu(host_a, host_b, host_c_cpu, N);
        double end_time = get_time();
        cpu_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_time / RUNS;

    printf("Benchmarking on GPU --> %d runs..... \n", RUNS);
    double gpu_time = 0.0;
    for (int i = 0; i < RUNS; i++)
    {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, THREADS_PER_BLOCK>>>(device_a, device_b, device_c, N);
        double end_time = get_time();
        gpu_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_time / RUNS;
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1e3);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1e3);
    printf("Speedup of GPU over CPU: %f\n", cpu_avg_time / gpu_avg_time); // For addition of vectors of 10 Million,
                                                                          // a speed-up of 1e4 is noticed.

    // Verify is both CPU and GPU operations yield the same result.
    cudaMemcpy(host_c_gpu, device_c, size_vec, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(host_c_cpu[i] - host_c_gpu[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    printf("The CPU and GPU results are %s\n", correct ? "matching" : "not matching");

    free(host_a);
    free(host_b);
    free(host_c_cpu);
    free(host_c_gpu);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}