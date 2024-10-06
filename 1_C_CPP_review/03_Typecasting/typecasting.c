#include <stdio.h>

int main()
{
    float pi = 3.14159;
    int int_pi = (int)pi;

    printf("Float pi: %f\tInt pi: %d\n", pi, int_pi);

    float val = 122.553;
    char char_val = (char)val;

    printf("Float val: %f\tChar val: %c\n", val, char_val);
}