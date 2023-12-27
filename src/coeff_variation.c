#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * It computes the population coefficient of variation of the elements of an array, 
 * ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The coefficient of variation of the non-NaN values in the array.
 */
double pop_cv(double *x, size_t n) {
    size_t i;
    double sum = 0.0;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for simd reduction(+ : nnan, res, sum)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            sum += x[i];
            res += x[i] * x[i];
        }
    }
    sum /= nnan;
    res = res / nnan - sum * sum;
    return sqrt(res) / sum;
}

/**
 * It computes the sample coefficient of variation of the elements of an array, 
 * ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The coefficient of variation of the non-NaN values in the array.
 */
double smp_cv(double *x, size_t n) {
    size_t i;
    double sum = 0.0;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for simd reduction(+ : nnan, res, sum)
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
    res *= nnan / (nnan - 1.0);
    return sqrt(res) / sum;
}

/* Testing function */
int main () {
    double x[5] = { 0.0, 0.0, 1.0, 2.1, 4.5};
    double y[50]= { 0.0, 0.0, 1.0, 2.1, 4.5, 0.0, 1.0, 2.1, -4.5, 8.1,
                    0.0, 0.0, -1.0, 2.1, 4.5, 0.0, -1.0, 2.1, 4.5, 8.1,
                    0.0, 0.0, 1.0, -2.1, 4.5, 0.1, -1.0, -2.1, 4.5, -8.1,
                    0.0, 0.0, -1.0, 2.1, -4.5, 0.2, 1.0, 2.1, 4.5, 8.1,
                    0.0, 0.0, 1.0, -2.1, -4.5, 0.3, 1.0, -2.1, -4.5, 8.1};
    printf("Population CV of x = %f\n", pop_cv(x, 5));
    printf("Sample CV of x = %f\n", smp_cv(x, 5));
    printf("Population CV of y = %f\n", pop_cv(y, 50));
    printf("Sample CV of y = %f\n", smp_cv(y, 50));
    return 0;
}
