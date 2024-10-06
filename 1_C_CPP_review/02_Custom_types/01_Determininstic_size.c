#include <stdio.h>

int main()
{
    int arr[] = {1, 2, 3, 4, 5, 6};

    size_t arr_size = sizeof(arr) / sizeof(arr[0]);
    printf("Size of array: %zu \n", arr_size);
    printf("Size of size_t: %zu \n", sizeof(size_t));
    printf("Size of int: %zu", sizeof(int));
    return 1;
}