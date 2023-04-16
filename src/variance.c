#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * It computes the population variance of the elements of an array, ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The variance of the non-NaN values in the array.
 */
double pop_variance(double *x, size_t n) {
    size_t i;
    double sum = 0.0;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res, sum)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            sum += x[i];
            res += x[i] * x[i];
        }
    }
    sum /= nnan;
    return res / nnan - sum * sum;
}

/**
 * It computes the sample variance of the elements of an array, ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The variance of the non-NaN values in the array.
 */
double smp_variance(double *x, size_t n) {
    size_t i;
    double sum = 0.0;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res, sum)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            sum += x[i];
            res += x[i] * x[i];
        }
    }
    sum /= nnan;
    res /= nnan;
    res -= sum * sum;
    return res * nnan / (nnan - 1.0);
}

/* Testing function */
int main () {
    double x[5] = { 0.0, 0.0, 1.0, 2.1, 4.5};
    printf("Population variance of x = %f\n", pop_variance(x, 5));
    printf("Sample variance of x = %f\n", smp_variance(x, 5));
    return 0;
}
