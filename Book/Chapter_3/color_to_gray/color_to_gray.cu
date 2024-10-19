#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3
__global__ void colortoGrayKernel(unsigned char *Pin, unsigned char *Pout, int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < height && col < width)
    {
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * CHANNELS;

        Pout[greyOffset] = 0.21f * Pin[rgbOffset] + 0.71f * Pin[rgbOffset + 1] + 0.07f * Pin[rgbOffset + 2];
    }
}

void check_results(const unsigned char *img_rgb, const unsigned char *img_gray, const int width, const int height)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int index = r * width + c;
            int rgb_index = CHANNELS * (index);

            unsigned char r = img_rgb[rgb_index];
            unsigned char g = img_rgb[rgb_index + 1];
            unsigned char b = img_rgb[rgb_index + 2];

            unsigned char gray_val = 0.21f * r + 0.71f * g + 0.07f * b;
            // if ((r != 0 || g != 0 || b != 0) && (gray_val == img_gray[r * width + c]))
            //     printf("@index: %d, (%d %d %d) gray_val: %d, img_gray_val: %d\n", index, r, g, b, (int)gray_val, (int)img_gray[r * width + c]);
            // // assert(gray_val == img_gray[r * width + c]);
        }
    }
}

int main()
{
    int width, height, num_channels;
    char path[] = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/testing/nasa-53884.jpg";
    char wpath[] = "/home/sri/CUDA/Programming-Massively-Parallel-Processors/testing/nasa-gray.jpg";

    unsigned char *rgb_h = stbi_load(path, &width, &height, &num_channels, CHANNELS);
    int rgb_bytes = sizeof(unsigned char) * width * height * CHANNELS;
    int gray_bytes = rgb_bytes / CHANNELS;

    unsigned char *gray_h, *gray_d, *rgb_d;
    gray_h = (unsigned char *)malloc(gray_bytes);

    cudaMalloc(&rgb_d, rgb_bytes);
    cudaMalloc(&gray_d, gray_bytes);

    cudaMemcpy(rgb_d, rgb_h, rgb_bytes, cudaMemcpyHostToDevice);

    int N_THREADS = 16;
    int GRID_N_X = (width + N_THREADS - 1) / N_THREADS;
    int GRID_N_Y = (height + N_THREADS - 1) / N_THREADS;

    dim3 BLOCK_DIM(N_THREADS, N_THREADS);
    dim3 GRID_DIM(GRID_N_X, GRID_N_Y);

    colortoGrayKernel<<<GRID_DIM, BLOCK_DIM>>>(rgb_d, gray_d, width, height);

    cudaMemcpy(gray_h, gray_d, gray_bytes, cudaMemcpyDeviceToHost);

    check_results(rgb_h, gray_h, width, height);

    stbi_write_jpg(wpath, width, height, 1, gray_h, 100);

    rgb_h = nullptr;
    free(gray_h);

    cudaFree(gray_d);
    cudaFree(rgb_d);

    printf("COMPLETED SUCCESSFULLY\n");

    return 0;
}