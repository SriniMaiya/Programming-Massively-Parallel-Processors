#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLUR_SIZE 7
#define CHANNELS 3

typedef struct
{
    int r;
    int g;
    int b;
} rgb_t;

__global__ void blurKernel(unsigned char *img_in, unsigned char *img_out, int width, int height)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < height)
    {
        rgb_t sum = {0, 0, 0};
        int nPixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow < height && curCol < width && curRow >= 0 && curCol >= 0)
                {
                    int red_index = CHANNELS * (curRow * width + curCol);
                    sum.r += img_in[red_index];
                    sum.g += img_in[red_index + 1];
                    sum.b += img_in[red_index + 2];
                    ++nPixels;
                }
            }
        }
        int red_index = CHANNELS * (row * width + col);
        img_out[red_index] = sum.r / nPixels;
        img_out[red_index + 1] = sum.g / nPixels;
        img_out[red_index + 2] = sum.b / nPixels;
    }
}

int main()
{
    int width = 0, height = 0, n_channels;
    unsigned char *img_b_h, *img_d, *img_b_d;
    char path[] = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/testing/nasa-53884.jpg";
    char wPath[] = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/testing/nasa-blurred.jpg";
    // char wPathtest[] = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/testing/50_50_rgb_read.jpg";

    unsigned char *img_h = stbi_load(path, &width, &height, &n_channels, CHANNELS);
    size_t img_bytes = sizeof(unsigned char) * width * height * CHANNELS;
    img_b_h = (unsigned char *)malloc(img_bytes);

    cudaMalloc(&img_d, img_bytes);
    cudaMalloc(&img_b_d, img_bytes);

    cudaMemcpy(img_d, img_h, img_bytes, cudaMemcpyHostToDevice);

    int N_THREADS = 16;
    int GRID_SIZE_X = (width + N_THREADS - 1) / N_THREADS;
    int GRID_SIZE_Y = (height + N_THREADS - 1) / N_THREADS;

    dim3 BLOCK_SIZE(N_THREADS, N_THREADS);
    dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);

    blurKernel<<<GRID_SIZE, BLOCK_SIZE>>>(img_d, img_b_d, width, height);

    cudaMemcpy(img_b_h, img_b_d, img_bytes, cudaMemcpyDeviceToHost);

    // stbi_write_jpg(wPathtest, width, height, CHANNELS, img_h, 100);
    stbi_write_jpg(wPath, width, height, CHANNELS, img_b_h, 100);

    img_h = nullptr;
    free(img_b_h);

    cudaFree(img_d);
    cudaFree(img_b_d);

    printf("COMPLETED SUCCESSFULLY");

    return 0;
}