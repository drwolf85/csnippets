#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Arithmetic averages */

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

void arithmetic_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        tmp = 1.0 / (double) (*n + 1);
        *avg = x * tmp + *avg * ((double) *n * tmp); 
        *n++;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Geometric averages */

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

void geometric_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        *avg = log(*avg);
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += log(x) * tmp;
        *avg = exp(*avg);
        *n++;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Harmonic averages */

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

void harmonic_average_online(double *avg, size_t *n, double x) {
    double tmp;
    if (*n > 0) {
        *avg = 1.0 / *avg;
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += tmp / x;
        *avg = 1.0 / *avg;
        *n++;
    }
    else {
        *n = 1;
        *avg = x;
    }
}

/* Generalized averages */

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

void generalized_average_online(double *avg, size_t *n, double x, double (*fun)(double), double (*invfun)(double)) {
    double tmp;
    if (*n > 0) {
        *avg = (*fun)(*avg);
        tmp = 1.0 / (double) (*n + 1);
        *avg *= (double) *n * tmp;
        *avg += (*fun)(x) * tmp;
        *avg = (*invfun)(*avg);
        *n++;
    }
    else {
        *n = 1;
        *avg = x;
    }
}
