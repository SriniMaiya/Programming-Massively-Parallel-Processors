#include <stdio.h>

int main()
{

    /*
    Voidpointer stores the memory addess of number (&number) as a void pointer (no data type).
    Voidpointer can't be directly dereferenced, so is cast to an integer pointer to store the integer value at the mem. addr.
                                                                                                                 (int*)void_intptr;
    The value can be then dereferenced by, *((int*)void_intptr)
    */
    int number = 42;
    void *void_intptr;

    void_intptr = &number;

    printf("Void Intptr address: %p\n", void_intptr);
    printf("Value at %p: %d\n\n", void_intptr, *((int *)void_intptr));

    float value = 3.1415;
    void *void_floatptr;

    void_floatptr = &value;
    printf("Void floatptr address: %p\n", void_floatptr);
    printf("Value at %p: %f\n", void_floatptr, *((float *)void_floatptr));

    return 0;
}