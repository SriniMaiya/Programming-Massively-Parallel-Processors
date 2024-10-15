#include <cublas_v2.h>
#include <iostream>
#include <cassert>
#include <cstdlib>
// nvcc -o cuBLAS_vectorAdd cuBLAS_vectorAdd.cu -lcublas

using std::cout, std::endl;

void init_vector(float *vec, const int N)
{
    for (int i = 0; i < N; i++)
    {
        vec[i] = rand() % 100;
    }
}

void check_results(const float *a, const float *b, const float *c, const int N)
{
    for (int i = 0; i < N; i++)
    {

        assert(c[i] == a[i] + b[i]);
    }
}

int main()
{
    int N = (int)1e7;
    int bytes = N * sizeof(float);

    // Host vectors.

    float *h_a, *h_b, *h_c;

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    init_vector(h_a, N);
    init_vector(h_b, N);

    // Device vectors.

    float *d_a, *d_b;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // ::: cuBLAS processing of vectors :::
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetVector(N, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(N, sizeof(float), h_b, 1, d_b, 1);

    const float scale = 1.0f;
    cublasSaxpy(handle, N, &scale, d_a, 1, d_b, 1);

    cublasGetVector(N, sizeof(float), d_b, 1, h_c, 1);

    check_results(h_a, h_b, h_c, N);

    cublasDestroy(handle);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(h_c);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "Error:  " << cudaGetErrorName(error) << cudaGetErrorString(error) << endl;
    }

    return 0;
}