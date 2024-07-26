#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40
#define MIN_EPS_QRRT 1e-10

/**
 * The function calculates the Hassanat distance between two vectors of double precision numbers.
 * 
 * @param x A pointer to an array of double values representing the coordinates of a point in
 * n-dimensional space.
 * @param y The parameter `y` is a pointer to an array of `double` values. It is used to 
 * calculate the Hassanat distance between two points represented by arrays `x` and `y`. 
 * @param n The parameter `n` represents the number of elements in the arrays `x` and `y`.
 * 
 * @return The function `hassanat_distance` returns the Hassanat distance between two points
 * represented by arrays `x` and `y` of length `n`.
 */
double hassanat_distance(double *x, double *y, size_t n) {
    double res = 0.0, num, den, mn, mx, amin;
    size_t i = 0;
    #pragma omp parallel for simd private(num, den, mn, mx, amin) reduction(+ : res)
    for (i = 0; i < n; i++) {
        num = fabs(x[i] - y[i]);
	mn = (double) (x[i] <= y[i]) * x[i] + (double) (x[i] > y[i]) * y[i];
	mx = (double) (x[i] >= y[i]) * x[i] + (double) (x[i] < y[i]) * y[i];
	amin = fabs(mn);
        den = 1.0 + mx + (double) (mn < 0.0) * amin;
        den += (double) (den < MIN_EPS_QRRT) * MIN_EPS;
        res += num / den;
    }
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8, 0.0};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6, 0.0};
    size_t i;
    printf("Hassanat distance between x and y (limited) is %f\n", hassanat_distance(x, y, 5));
    printf("Hassanat distance between x and y (full) is %f\n", hassanat_distance(x, y, 6));
    return 0;
}
