#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_ITER 100

/**
@brief Residuals of a weighted linear model (based on OLS)
@param res empty vector to store the residuals (in output)
@param py response vector (example data for model output)
@param w vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
static inline void wlm_resid(double *res, double *py, double *w, double *pdta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *y, *dta, *iw, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    dta = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    y = (double *) malloc(dim[0] * sizeof(double));
    iw = (double *) malloc(dim[0] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && y && dta && iw && vec) {
        /* Adjust data for the weights */
        for (i = 0; i < dim[0]; i++) {
            iw[i] = sqrt(w[i]);
            y[i] = py[i] * iw[i];
            #pragma omp for simd
            for (j = 0; j < dim[1]; j++) {
                dta[*dim * j + i] = pdta[*dim * j + i] * iw[i];
            }
            iw[i] = 1.0 / iw[i];
        }
        /* Computing matrix Q */
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
        /* Computing least square residuals (Q^t W^{0.5} y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            vec[k] = tmp;
        }
        /* Computing least square residuals y - W^{-0.5} (QQ^t) W^{0.5} y */
        for (j = 0; j < dim[0]; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = y[j] - tmp;
            res[j] *= iw[j];
        }
    }
    free(q);
    free(y);
    free(iw);
    free(vec);
    free(dta);
}

/**
@brief Weighted iterated least absolute residuals
@param res empty vector to store the residulas
@param py response vector (example data for model output)
@param pw vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void wilar_resid(double *res, double *py, double *pw, double *pdta, int *dim) {
    int i, j, k = 0;
    double *w;

    w = (double *) malloc(dim[0] * sizeof(double));
    /* Get OLS residuals for a weighted linear model */
    wlm_resid(res, py, pw, pdta, dim);
    if (w) do { /* Get WOLS residuals (iteratively) */
        #pragma omp for simd
        for (i = 0; i < dim[0]; i++) { /* Compute the weights */
            w[i] = pw[i] / fabs(res[i]);
            w[i] = isfinite(w[i]) ? w[i] : pw[i];
        }
        wlm_resid(res, py, w, pdta, dim);
        k++;
    }
    while (k < MAX_ITER);
    free(w);
}

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// w <- runif(n, 0.25, 4)
// dyn.load("test.so")
// b <- .C("wilar_resid", res = double(n), y, w, X, dim(X), DUP = FALSE)$res
// print(b, digit = 22)
