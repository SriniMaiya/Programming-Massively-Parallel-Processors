#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Initialize the pointer to NULL
    int *ptr = NULL;
    printf("1. Initial pointer value: %p\n", (void *)ptr);

    // Check the null pointer before using
    if (ptr == NULL)
    {
        printf("2. ptr is NULL, cannot dereference\n");
    }

    ptr = malloc(sizeof(int));
    if (ptr == NULL)
    {
        printf("3. Malloc failed!!\n");
        return -1;
    }

    printf("4. After allocation, ptr value: %p\n", (void *)ptr);

    // Safe to use the pointer after null check
    *ptr = 42;
    printf("5. Value at pointer: %d\n", *ptr);

    // Remove the pointer.
    free(ptr);
    ptr = NULL;

    printf("6. After freeing, ptr value: %p\n", (void *)ptr);

    if (ptr == NULL)
    {
        printf("7. ptr is NULL, avoided use of pointer after free.");
    }

    return 0;
}