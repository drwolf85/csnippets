#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the Manhattan distance between two vectors of double precision numbers.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values. It is used in the
 * `manhattan_distance` function to calculate the Manhattan distance between two points represented by
 * arrays `x` and `y`. The function calculates the absolute difference between each corresponding
 * element of `x`
 * @param n The parameter `n` represents the number of elements in the arrays `x` and `y`.
 * 
 * @return The function `manhattan_distance` returns the Manhattan distance between two points
 * represented by arrays `x` and `y` of length `n`.
 */
double manhattan_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    size_t i = 0;
    #pragma omp parallel for simd reduction(+ : res)
    for (i = 0; i < n; i++) {
        res += fabs(x[i] - y[i]);
    }
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6};
    size_t i;
    printf("Manhattan distance between x and y is %f\n", manhattan_distance(x, y, 5));
    return 0;
}
