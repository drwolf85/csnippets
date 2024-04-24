#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * A `dta_vec` is a structure with two double-precision floating point numbers, `v` and `w`.
 * @property {double} v - The value of the item
 * @property {double} w - The weight of the item
 */
typedef struct dta_vec {
    double v; /* Values */
    double w; /* Weights */
} dta_vec;

/**
 * @brief Comparison function for double values
 *
 * @param aa Pointer to double
 * @param bb Pointer to double
 * @return int
 */
int cmp_double(void const *aa, void const *bb) {
    double a = *(double *)aa;
    double b = *(double *)bb;
    if (isnan(a)) return 1;
    if (isnan(b)) return -1;
    return (int) (a >= b) * 2 - 1;
}

/**
 * @brief Standard median algorith (based on sorting)
 *
 * @param x Vector of data points
 * @param n Size of the vector
 * @return double
 */
double median(double *x, size_t n) {
    size_t i;
    double res, *y;
    y = (double *)malloc(n * sizeof(double));
    if (y) {
        memcpy(y, x, n * sizeof(double));
        qsort(y, n, sizeof(double), cmp_double);
        i = n >> 1;
        res = y[i] * 0.5 + y[i - (size_t)!(n & 1)] * 0.5;
    }
    free(y);
    return res;
}

/**
 * It computes the weighted median of a set of numbers by minimizing 
 * the weighted mean absolute error
 * 
 * @param x the array of values
 * @param w the weights of the data points
 * @param n the number of elements in the array
 * 
 * @return The weighted median of the array `x`.
 */
double weighted_median(double *x, double *w, size_t n) {
    size_t i;
    double res = 0.0;
    double sum, nc, *e;
    e = (double *)malloc(n * sizeof(double));
    if (e) {
        sum = 0.0;
        nc = 0.0;
        for (i = 0; i < n; i++) {
            if (isnan(w[i]) || isnan(x[i])) continue;
            sum += w[i] * x[i];
            nc += w[i];
        }
        sum /= nc;
        do {
            res = sum;
            for (i = 0; i < n; i++) {
                if (isnan(w[i]) || isnan(x[i])) continue;
                sum = fabs(x[i] - res);
                e[i] = w[i] / (sum < 1e-9 ? 1e-9 : sum);
            }
            sum = 0.0;
            nc = 0.0;
            for (i = 0; i < n; i++) {
                if (isnan(w[i]) || isnan(x[i])) continue;
                sum += e[i] * x[i];
                nc += e[i];
            }
            sum /= nc;
        } while(fabs(res - sum) > 1e-9);
        res = sum;
    }
    free(e);
    return res;
}

/**
 * A vector is a structure with three fields: `v`, `w`, and `i`, 
 * where `v` and `w` are doubles and `i` is a size_t.
 * @property {double} v - the value in the vector
 * @property {double} w - the weight of the value
 * @property {size_t} i - the index of the entry in the original array
 */
typedef struct vector {
    double v;
    double w;
    size_t i;
} vector;


/**
 * It compares two vectors, and returns -1 if the first is less than the 
 * second, 0 if they're equal, and 1 if the first is greater than the second
 * 
 * @param aa the first vector to compare
 * @param bb the second vector to compare
 * 
 * @return -1, 0, or 1
 */
int cmp_vector(void const *aa, void const *bb) {
    vector a = *(vector *) aa;
    vector b = *(vector *) bb;
    return 2 * (int)((a.v >= b.v) || (isnan(a.v) && !isnan(b.v))) - 1;
}

/**
 * It takes a vector of weights and a vector of values, and returns the weighted median of the values
 * 
 * @param x the array of values
 * @param w the weights of each observation
 * @param n the number of data points
 * 
 * @return The median of the data.
 */
double wt_med_hst(double *x, double *w, size_t n) {
    size_t i, idx, count = 0;
    size_t N_BINS = (size_t) sqrt((double) n);
    double wts[N_BINS];
    double u, v, ttwt = *w;
    double range, cdf;
    u = *x;
    v = u;

    for (i = 1; i < n; i++) {
        u += (double) (x[i] > u) * (x[i] - u);
        v += (double) (x[i] < v) * (x[i] - v);
        ttwt += w[i];
    }
    ttwt = 1.0 / ttwt;
    do {
        range = u - v;
        range = (double) N_BINS / range;
        memset(wts, 0, N_BINS * sizeof(double));
        for (i = 0; i < n; i++) {
            idx = (size_t) (range * (x[i] - v) * (double) (x[i] > v));
            idx -= (size_t) (idx >= N_BINS) * (idx - N_BINS + 1);
            wts[idx] += w[i] * ttwt;
        }
        cdf = 0.0;
        for (i = 0; cdf < 0.5 && i < N_BINS; i++) {
            cdf += wts[i];
        }
        i -= (size_t) (i >= 1);
        range = 1.0 / range;
        v += (double) i * range;
        u = v + range;
    } while (range > 1e-16);
    return v;
}

/**
 * It sorts the data by value, then it adds up the weights until it reaches the median weight, 
 * and returns the value at that point
 * 
 * @param x the array of values
 * @param w the weights of the observations
 * @param n the number of elements in the array
 * 
 * @return The median of the weighted sample.
 */
double wt_med_std(double *x, double *w, size_t n) {
    vector *a;
    size_t i, nn = 0;
    double res = 0.0 / 0.0;
    double sum = 0.0;
    bool b;

    a = (vector *) malloc(n * sizeof(vector));
    if (a && n > 0) {
        for (i = 0; i < n; i++) {
            a[i].v = x[i];
            a[i].w = w[i];
            a[i].i = i;
            b = !isnan(x[i]);
            sum += w[i] * (double) b;
            nn += (size_t) b;
        }
        sum *= 0.5;
        qsort(a, n, sizeof(vector), cmp_vector);
        res = a[0].v;
        for (i = 1; i < nn && a[i - 1].w < sum; i++) {
            a[i].w += a[i - 1].w;
            res = a[i].v;
        }
    }
    free(a);
    return res;
}
