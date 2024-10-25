#include <stdio.h>
#include <stdlib.h>

#define N 256
#define BOTTLENECK 32

void init_array(int *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        array[i] = i;
    }
}

int main()
{
    int bytes = Width_N * sizeof(int);

    int *array = (int *)malloc(bytes);

    init_array(array, Width_N);

    // ::: Simulating :::
    // thread index: 0, 1, 2,...63
    // BOTTLENECK: blockDim.x * gridDim.x = 32 ==> Our GPU can only process max of 32 elements at once.

    for (int thread_index = 0; thread_index < 1000; thread_index++) // Not accurate description of thread creation.
                                                                    // In CUDA CPP the threads are created dynamically.
    {
        printf("THREAD INDEX: %d", thread_index);
        for (int i = thread_index; i < Width_N / 4; i += BOTTLENECK)
        {
            int4 *int4_array = (int4 *)array;
            int4 indexed_element = int4_array[i];
            printf("\t@ index %d, indexed values are: {%d, %d, %d, %d} \n",
                   i, indexed_element.x, indexed_element.y, indexed_element.z, indexed_element.w);
        }
    }
}