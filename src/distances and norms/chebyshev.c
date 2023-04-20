#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the Chebyshev distance between two vectors of the same length.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values representing the coordinates
 * of a point in n-dimensional space.
 * @param n The parameter `n` represents the number of dimensions in the space in which the Chebyshev
 * distance is being calculated. In other words, it is the number of elements in the input arrays `x`
 * and `y`.
 * 
 * @return The function `chebyshev_distance` returns the maximum absolute difference between
 * corresponding elements of two arrays `x` and `y` of length `n`. This is also known as the Chebyshev
 * distance or the L-infinity distance.
 */
double chebyshev_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    double tmp;
    size_t i = 0;
    tmp = fabs(x[i] - y[i]);
    #pragma omp parallel for simd reduction(max : res)
    for (i = 1; i < n; i++) {
        tmp = fabs(x[i] - y[i]);
        res = tmp > res ? tmp : res;
    }
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6};
    size_t i;
    printf("Chebyshev distance between x and y is %f\n", chebyshev_distance(x, y, 5));
    return 0;
}
