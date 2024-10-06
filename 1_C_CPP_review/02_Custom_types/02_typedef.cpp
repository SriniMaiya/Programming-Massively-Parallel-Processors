#include <iostream>

typedef struct Point
{
    float x;
    float y;
} point_t;

int main()
{
    point_t pt = {3, 5};
    printf("Size of struct: %zu\n", sizeof(point_t));
    // Output: 8 bytes = 4 bytes (float x) + 4 bytes (float y)
    return 0;
}