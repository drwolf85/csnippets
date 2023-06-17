#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_ITER 100

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
@brief Residuals of a linear model (based on OLS)
@param res empty vector to store the residuals (in output)
@param y response vector (example data for model output)
@param dta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void lm_resid(double *res, double *y, double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && vec) {
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
                tmp /= v;
                #pragma omp for simd
                for (j = 0; j < dim[0]; j++) 
                    q[*dim * i + j] -= tmp * q[*dim * k + j];
            }
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            itmp = 1.0 / sqrt(tmp);
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *= itmp;
        }
        /* Computing least square residuals (Q^t y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            vec[k] = tmp;
        }
        /* Computing least square residuals y - (QQ^t) y */
        for (j = 0; j < dim[0]; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = y[j] - tmp;
        }
    }
    free(q);
    free(vec);
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

/**
@brief Iterated least absolute residuals
@param res empty vector to store the residulas
@param py response vector (example data for model output)
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void ilar_resid(double *res, double *py, double *pdta, int *dim) {
    int i, j, k = 0;
    double *w, *coef;

    w = (double *) malloc(dim[0] * sizeof(double));
    coef = (double *) malloc(dim[1] * sizeof(double));
    /* Get OLS residuals for a linear model */
    lm_resid(res, py, pdta, dim);
    if (coef && w) do { /* Get WOLS residuals (iteratively) */
        #pragma omp for simd
        for (i = 0; i < dim[0]; i++) { /* Compute the weights */
            w[i] = 1.0 / fabs(res[i]);
            w[i] = isfinite(w[i]) ? w[i] : 1.0;
        }
        wlm_coef(coef, py, w, pdta, dim);
        for (i = 0; i < dim[0]; i++) { /* Compute the residuals */
            res[i] = 0.0;
            for (j = 0; j < dim[1]; j++) {
                res[i] -= pdta[*dim * j + i] * coef[j];
            }
            res[i] += py[i];
        }
        k++;
    }
    while (k < MAX_ITER);
    free(w);
    free(coef);
}

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// dyn.load("test.so")
// b <- .C("ilar_resid", res = double(n), y, X, dim(X), DUP = FALSE)$res
// print(b, digit = 22)
