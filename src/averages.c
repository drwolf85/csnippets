#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Arithmetic averages */

/**
 * It computes the arithmetic average of the elements of an array, ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The arithmetic average of the non-NaN values in the array.
 */
double arithmetic_average(double *x, size_t n) {
    size_t i;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            res += x[i];
        }
    }
    return res / nnan;
}

/**
 * It calculates the arithmetic average of a set of numbers.
 * 
 * @param avg the average value (previously computed)
 * @param n the number of samples (previously used)
 * @param x the new value to be added to the average
 */
void arithmetic_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        tmp = 1.0 / (double) (*n + 1);
        *avg = x * tmp + *avg * ((double) *n * tmp); 
        *n += 1;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Geometric averages */

/**
 * It computes the geometric average of an array of doubles, ignoring NaN values
 * 
 * @param x the array of numbers
 * @param n the number of elements in the array
 * 
 * @return The geometric average of the array `x`.
 */
double geometric_average(double *x, size_t n) {
    size_t i;
    double res = 1.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            res *= x[i];
        }
    }
    return pow(res, 1.0 / nnan);
}

/**
 * "For each element in the array, if it's not NaN, add one to the number of non-NaN elements, 
 * and add the log of the element to the sum of the logs of the elements."
 * 
 * The first thing to notice is that the function is almost identical to the previous one.
 * 
 * @param x the array of numbers
 * @param n the number of elements in the array
 * 
 * @return The geometric mean of the array `x`.
 */
double geometric_average2(double *x, size_t n) {
    size_t i;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            res += log(x[i]);
        }
    }
    return exp(res / nnan);
}

/**
 * It calculates the geometric average of a set of numbers.
 * 
 * @param avg the current average (previoulsy computed)
 * @param n the number of samples seen so far
 * @param x the new value to be added to the average
 */
void geometric_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        *avg = log(*avg);
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += log(x) * tmp;
        *avg = exp(*avg);
        *n += 1;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Harmonic averages */

/**
 * It computes the harmonic average of the elements of an array, ignoring NaN values
 * 
 * @param x the array of values
 * @param n the number of elements in the array
 * 
 * @return The harmonic mean of the array.
 */
double harmonic_average(double *x, size_t n) {
    size_t i;
    double res = 1.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            res += 1.0 / x[i];
        }
    }
    return nnan / res;
}

/**
 * It calculates the harmonic average of a set of numbers.
 * 
 * @param avg the current average (previously computed)
 * @param n the number of samples (previously used)
 * @param x the new value to be "added" to the average
 */
void harmonic_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        *avg = 1.0 / *avg;
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += tmp / x;
        *avg = 1.0 / *avg;
        *n += 1;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Generalized averages */

/**
 * It computes the generalized average of the elements of an array, using a 
 * function `fun` and its inverse (i.e., `invfun`). This function ignores 
 * the NaN values
 *
 * @param x the array of values
 * @param n the number of elements in the array
 * @param fun a function that takes a double and returns a double
 * @param invfun the inverse of the function to be applied to the data
 * 
 * @return The average of the function applied to the elements of the array.
 */
double generalized_average(double *x, size_t n, double (*fun)(double), double (*invfun)(double)) {
    size_t i;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for private(i) reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            nnan += 1.0;
            res += (*fun)(x[i]);
        }
    }
    return (*invfun)(res / nnan);
}

/**
 * It takes a running average of a function of a variable
 * 
 * @param avg the average value (previously computed)
 * @param n the number of samples seen so far
 * @param x the new value to be added to the average
 * @param fun a function that takes a double and returns a double
 * @param invfun the inverse of the function you want to use
 */
void generalized_average_online(double *avg, size_t *n, double x, double (*fun)(double), double (*invfun)(double)) {
    double tmp;
    if (*n > 0) {
        *avg = (*fun)(*avg);
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += (*fun)(x) * tmp;
        *avg = (*invfun)(*avg);
        *n += 1;
    }
    else {
        *n = 1;
        *avg = x;
    }
}
