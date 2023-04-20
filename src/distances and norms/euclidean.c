#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the Euclidean distance between two vectors of double values using OpenMP
 * parallelization.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values. The function
 * `euclidean_distance` calculates the Euclidean distance between two arrays `x` and `y` of the same
 * length `n`, where `x` is not explicitly given as a parameter but is assumed
 * @param n The parameter `n` represents the number of dimensions in the Euclidean space. In other
 * words, it is the number of elements in the input arrays `x` and `y`.
 * 
 * @return The function `euclidean_distance` returns the Euclidean distance between two points
 * represented as arrays of `double` values.
 */
double euclidean_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    double tmp;
    size_t i = 0;
    #pragma omp parallel for simd private(tmp) reduction(+ : res)
    for (i = 0; i < n; i++) {
        tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    res = sqrt(res);
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6};
    size_t i;
    printf("Euclidean distance between x and y is %f\n", euclidean_distance(x, y, 5));
    return 0;
}
