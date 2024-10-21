#include <stdio.h>
#include <assert.h>

void print_mat(float *, int, int);
__global__ void matVecMult(float *vec_out, float *mat, float *vec_in, int height, int width)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < height)
    {
        float sum = 0;
        for (int col = 0; col < width; col++)
        {
            sum += mat[row * width + col] * vec_in[col];
        }
        vec_out[row] = sum;
    }
}

void init_mat(float *mat, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

void check_results(float *vec_in, float *mat, float *vec_out, int height, int width)
{
    // printf("::: M :::\n");
    // print_mat(mat, height, width);
    // printf("::: Vec_in :::\n");
    // print_mat(vec_in, width, 1);
    // printf("::: Vec_out :::\n");
    // print_mat(vec_out, height, 1);
    // printf("\n");
    for (int row = 0; row < height; row++)
    {
        float sum = 0;
        for (int col = 0; col < width; col++)
        {
            sum += mat[row * width + col] * vec_in[col];
        }
        printf("%f == %f: %d\n", sum, vec_out[row], abs(sum - vec_out[row]) < 1e-4);
        assert(abs(sum - vec_out[row]) < 1e-3);
    }
    printf("\n");
}

void print_mat(float *mat, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            printf("%.3f ", mat[r * cols + c]);
        }
        printf("\n");
    }
}

int main()
{
    float *M_d, *V_in_d, *V_out_d, *M_h, *V_in_h, *V_out_h;
    int height = 1024; // height of m
    int width = 4096;  // width of n

    size_t bytes_M = sizeof(float) * height * width;
    size_t bytes_Vin = sizeof(float) * width;
    size_t bytes_Vout = sizeof(float) * height;

    M_h = (float *)malloc(bytes_M);
    V_in_h = (float *)malloc(bytes_Vin);
    V_out_h = (float *)malloc(bytes_Vout);

    srand(42);
    init_mat(M_h, height * width);
    init_mat(V_in_h, width);
    cudaMalloc(&M_d, bytes_M);
    cudaMalloc(&V_in_d, bytes_Vin);
    cudaMalloc(&V_out_d, bytes_Vout);

    cudaMemcpy(M_d, M_h, bytes_M, cudaMemcpyHostToDevice);
    cudaMemcpy(V_in_d, V_in_h, bytes_Vin, cudaMemcpyHostToDevice);

    int THREADS = 8;
    // int GRID_SIZE_X = (width_n + THREADS - 1) / THREADS;
    int GRID_SIZE_Y = (height + THREADS - 1) / THREADS;

    dim3 BLOCK_SIZE(1, THREADS);
    dim3 GRID_SIZE(1, GRID_SIZE_Y);

    matVecMult<<<GRID_SIZE, BLOCK_SIZE>>>(V_out_d, M_d, V_in_d, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(V_out_h, V_out_d, bytes_Vout, cudaMemcpyDeviceToHost);

    check_results(V_in_h, M_h, V_out_h, height, width);

    free(M_h);
    free(V_in_h);
    free(V_out_h);
    cudaFree(M_d);
    cudaFree(V_in_d);
    cudaFree(V_out_d);

    printf("COMPLETED SUCCESSFULLY\n");
    return 0;
}