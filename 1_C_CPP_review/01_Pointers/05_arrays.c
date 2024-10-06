#include <stdio.h>

int main()
{
    int arr[] = {42, 12, 14, 124, 421};

    int *ptr = arr; // ptr points to the first element of the arr.
    printf("Position one: %d\n\n", *ptr);

    for (int i = 0; i < 5; i++)
    {
        printf("arr[%d]: %d\t@:%p\n", i, *ptr, ptr);
        ptr++;
    }

    return 0;
}
