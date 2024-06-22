#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40

/**
 * The function calculates the Canberra distance between two vectors of double precision numbers.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values. It is used to 
 * calculate the Canberra distance between two points represented by arrays `x` and `y`. 
 * @param n The parameter `n` represents the number of elements in the arrays `x` and `y`.
 * 
 * @return The function `Canberra_distance` returns the Canberra distance between two points
 * represented by arrays `x` and `y` of length `n`.
 */
double canberra_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    size_t i = 0;
    #pragma omp parallel for simd reduction(+ : res)
    for (i = 0; i < n; i++) {
        res += fabs(x[i] - y[i]) / (fabs(x[i]) + fabs(y[i]) + MIN_EPS);
    }
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8, 0.0};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6, 0.0};
    size_t i;
    printf("Canberra distance between x and y (limited) is %f\n", canberra_distance(x, y, 5));
    printf("Canberra distance between x and y (full) is %f\n", canberra_distance(x, y, 6));
    return 0;
}
