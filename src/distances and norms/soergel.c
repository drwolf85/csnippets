#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40

/**
 * The function calculates the Soergel distance (also known as the Ruzieka distance) between
 * two vectors of double precision numbers.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values. It is used to 
 * calculate the Soergel distance between two points represented by arrays `x` and `y`. 
 * @param n The parameter `n` represents the number of elements in the arrays `x` and `y`.
 * 
 * @return The function `soergel_distance` returns the Soergel distance between two points
 * represented by arrays `x` and `y` of length `n`.
 */
double soergel_distance(double *x, double *y, size_t n) {
    double num = 0.0, den = MIN_EPS, ax, ay;
    size_t i = 0;
    #pragma omp parallel for simd private(ax, ay) reduction(+ : num, den)
    for (i = 0; i < n; i++) {
        ax = fabs(x[i]);
        ay = fabs(y[i]);
        num += fabs(x[i] - y[i]);
        den += (double) (ax >= ay) * ax + (double) (ax < ay) * ay;
    }
    return num / den;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8, 0.0};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6, 0.0};
    size_t i;
    printf("Soergel distance between x and y (limited) is %f\n", soergel_distance(x, y, 5));
    printf("Soergel distance between x and y (full) is %f\n", soergel_distance(x, y, 6));
    return 0;
}
