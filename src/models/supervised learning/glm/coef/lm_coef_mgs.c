#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
 * The function `inverseUT` takes a square matrix `mat` of dimension `n` and returns the inverse of the
 * upper triangular matrix `mat`
 * 
 * @param mat the matrix to be inverted
 * @param nn the number of rows and columns in the matrix
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
 * The function `lm_coef` computes the regression coefficients of a linear model using the QR
 * decomposition of the design matrix
 * 
 * @param coef the coefficients of the regression
 * @param y the response variable
 * @param dta the data matrix, with each row being a sample and each column being a feature.
 * @param dim a vector of length 2, where `dim[0]` is the number of observations and `dim[1]` is the number
 * of variables.
 */
void lm_coef(double *coef, double *y, double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q,*r;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    r = (double *) calloc(dim[1] * dim[1], sizeof(double));
    memset(coef, 0, dim[1] * sizeof(double));
    if (q && r) {
        memcpy(q, dta, dim[0] * dim[1] * sizeof(double));
        for (i = 0; i < dim[1]; i++) {
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            tmp = sqrt(tmp);
            r[dim[1] * i + i] = tmp;
            tmp = 1.0 / tmp;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *= tmp;
            /* Orthogonalization */
            for (j = i + 1; j < dim[1]; j++) {
                for (k = 0; k < dim[0]; k++)
                    r[dim[1] * j + i] += q[*dim * i + k] * q[*dim * j + k];
                for (k = 0; k < dim[0]; k++)
                    q[*dim * j + k] -= q[*dim * i + k] * r[dim[1] * j + i];
            }
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
}

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// dyn.load("test.so")
// b <- .C("lm_coef", coef = double(p), y, X, dim(X), DUP = FALSE)$coef
// print(lm(y ~ 0 + X)$coef, digit = 22)
// print(b, digit = 22)
