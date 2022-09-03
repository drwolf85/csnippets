#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
@brief Inverse of an Upper Triangular Matrix
@param mat upper triangular matrix (in column major format)
@param nn number of rows in the matrix `mat` 
*/
void inverseUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        pos = (n + 1) * (i - 1);
        mat[pos] = 1.0 / mat[pos];
        for (j = n - 1; j + 1 > i; j--) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * j + k] * mat[n * k + i - 1];
            }
            mat[n * j + i - 1] = tmp * (- mat[pos]);
        }
    }
}

/**
@brief Weighted Linear Regression
@param coef empty vector to store the output
@param py response vector (example data for model output)
@param w vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void wlm_coef(double *coef, double *py, double *w, double *pdta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *r, *y, *dta;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    r = (double *) calloc(dim[1] * dim[1], sizeof(double));
    dta = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    y = (double *) malloc(dim[0] * sizeof(double));
    memset(coef, 0, dim[1] * sizeof(double));
    if (q && r && y && dta) {
        /* Adjust data for the weights */
        for (i = 0; i < dim[0]; i++) {
            tmp = sqrt(w[i]);
            y[i] = py[i] * tmp;
            #pragma omp for simd
            for (j = 0; j < dim[1]; j++) {
                dta[*dim * j + i] = pdta[*dim * j + i] * tmp;
            }
        }
        /* Computing matrix Q (and R) */
        for (i = 0; i < dim[1]; i++) {
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] = dta[*dim * i + j];
            for (k = 0; k < i; k++) {
                tmp = 0.0;
                v = 0.0;
                for (j = 0; j < dim[0]; j++) {
                    tmp += q[*dim * k + j] * dta[*dim * i + j];
                    v += q[*dim * k + j] * q[*dim * k + j];
                }
                r[dim[1] * i + k] = tmp / sqrt(v); /* Matrix R */
                tmp /= v;
                #pragma omp for simd
                for (j = 0; j < dim[0]; j++) 
                    q[*dim * i + j] -= tmp * q[*dim * k + j];
            }
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            tmp = sqrt(tmp);
            r[dim[1] * i + i] = tmp; /* Matrix R */
            itmp = 1.0 / tmp;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *=itmp;
        }
        /* Invert matrix R */
        inverseUT(r, &dim[1]);
        /* Computing regression coefficients */
        for (i = dim[1]; i > 0; i--) {
            k = i - 1;
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            #pragma omp for simd
            for (j = 0; j < i; j++) {
                coef[j] += r[dim[1] * k + j] * tmp;
            }
        }
    }
    free(r);
    free(q);
    free(y);
    free(dta);
}

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// w <- runif(n, 0.25, 4)
// dyn.load("test.so")
// b <- .C("wlm_coef", coef = double(p), y, w, X, dim(X), DUP = FALSE)$coef
// print(lm(y ~ 0 + X, weights = w)$coef, digit = 22)
// print(b, digit = 22)
