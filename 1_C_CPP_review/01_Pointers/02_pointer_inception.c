#include <stdio.h>

int main()
{
    int value = 42;
    int *pointer = &value;
    int **pointer2 = &pointer;
    int ***pointer3 = &pointer2;

    printf("Address of pointer: %p\n", pointer);
    printf("Address of pointer2: %p\n", pointer2);
    printf("Address of pointer3: %p\n", pointer3);
    printf("Dereferenced pointer3: %d\n", ***pointer3);
    return 0;
}