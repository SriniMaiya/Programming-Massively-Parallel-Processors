#include <stdio.h>

int main()
{
    int x = 10;
    int *ptr = &x;
    printf("Addess of x: %p\n", ptr);
    printf("Value of x: %d\n", *ptr);
    return 0;
}