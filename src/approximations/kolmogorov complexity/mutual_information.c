#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/**
 * @brief Normalized mutual information
 * 
 * @param x Pointer to a vector of data x
 * @param y Pointer to a vector of data y
 * @param n Length of the two vectors
 * @return double 
 */
double nmi(double *x, double *y, size_t n) {
    size_t i, j, k;
    size_t nb = (size_t) sqrt(0.25 * (double) n);
    double const inb = 1.0 / (double) nb;
    size_t *fhat = calloc(sizeof(size_t), nb * nb);
    size_t *fx = calloc(sizeof(size_t), nb);
    size_t *fy = calloc(sizeof(size_t), nb);
    double xmn, xmx, ymn, ymx, xs, ys;
    double tmp, rx, ry, res = nan("");

    if (fx && fy && fhat && n > 64) {
        /* Get range of x and y */
        xmn = xmx = *x;
        ymn = ymx = *y;
        #pragma omp for simd
        for (i = 1; i < n; i++) {
            xmn += (xmn > x[i]) * (x[i] - xmn);
            xmx += (xmx < x[i]) * (x[i] - xmx);
            ymn += (ymn > y[i]) * (y[i] - ymn);
            ymx += (ymx < y[i]) * (y[i] - ymx);
        }
        xs = (double) nb / (xmx - xmn);
        ys = (double) nb / (ymx - ymn);
        /* Compute histograms */
        for (i = 0; i < n; i++) {
            j = (size_t) ((x[i] - xmn) * xs);
            j += (j >= nb) * (nb - j - 1);
            ++fx[j];
            k = (size_t) ((y[i] - ymn) * ys);
            k += (k >= nb) * (nb - k - 1);
            ++fy[k];
            j += nb * k;
            ++fhat[j];
        }
        res = 0.0;
        rx = 0.0;
        ry = 0.0;
        for (i = 0; i < nb; i++) {
            tmp = (double) fx[i] / (double) n;
            rx -= tmp > 0.0 && tmp < 1.0 ? tmp * log(tmp) : 0.0;
            tmp = (double) fy[i] / (double) n;
            ry -= tmp > 0.0 && tmp < 1.0 ? tmp * log(tmp) : 0.0;
            tmp = (double) fhat[i] / (double) n;
            res -= tmp > 0.0 && tmp < 1.0 ? tmp * log(tmp) : 0.0;
        }
        for (i = nb; i < nb * nb; i++) {
            tmp = (double) fhat[i] / (double) n;
            res -= tmp > 0.0 && tmp < 1.0 ? tmp * log(tmp) : 0.0;
        }
        res = rx + ry - res;
        res /= rx < ry ? rx : ry;
    }
    free(fhat);
    free(fx);
    free(fy);
    return res;
}

/* Test function */

int main () {
    double res, approx;
    #include "../../.data/linear_data.h"
    res = nmi(x0, y, N);
    approx = nmi(x1, y, N);
    printf("Mutual information when y is dependent on x0 "
           "and independet from x1:\n");
    printf("\tx0-vs-y: %f\n", res);
    printf("\tx1-vs-y: %f\n", approx);
    // printf("Reverse:\n");
    // res = nmi(y, x0, N);
    // approx = nmi(y, x1, N);
    // printf("\ty-vs-x0: %f\n", res);
    // printf("\ty-vs-x1: %f\n", approx);
    return 0;
}