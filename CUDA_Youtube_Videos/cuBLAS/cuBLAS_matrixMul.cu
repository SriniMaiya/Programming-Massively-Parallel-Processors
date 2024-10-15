#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <curand.h>
#include <assert.h>
#include <time.h>
#include <math.h>
// nvcc -o cuBLAS_matrixMul cuBLAS_matrixMul.cu -lcublas -lcurand
// cuBLAS is column major.
void verify_solution(const float *a, const float *b, const float *c, const int n)
{
    float temp;
    float delta = 1e-3;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            temp = 0.0f;
            for (int k = 0; k < n; k++)
            {
                temp += a[k * n + i] * b[j * n + k];
            }
            assert(fabs(temp - c[j * n + i]) < delta);
        }
    }
}

int main()
{
    int n = 1 << 10;
    int bytes = n * n * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Pseudo random number generator: Handle & generator.
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed.
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Fill the matrix with random number directly on the device.
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scaling factors.
    float alpha = 1.0f;
    float beta = 0.0f;

    // calculate c = (alpha * a) * b + (beta * c)
    // (m X n) * (n X k) = (m X k)
    // lda: leading dimension of A
    // ldb: leading dimension of B
    // ldb: leading dimension of C

    // Signature: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    cublasGetMatrix(n, n, sizeof(float), d_a, n, h_a, n);
    cublasGetMatrix(n, n, sizeof(float), d_b, n, h_b, n);
    cublasGetMatrix(n, n, sizeof(float), d_c, n, h_c, n);
    // cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    verify_solution(h_a, h_b, h_c, n);

    printf("COMPLETED SUCCESSFULLY");

    return 0;
}