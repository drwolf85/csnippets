#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
 * @brief Residuals of a linear model (based on OLS)
 * @param res empty vector to store the residuals (in output)
 * @param y response vector (example data for model output)
 * @param dta matrix of data (column-major format)
 * @param dim vector of dimension of `dta` matrix 
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

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// dyn.load("test.so")
// b <- .C("lm_resid", res = double(n), y, X, dim(X), DUP = FALSE)$res
// print(err <- resid(lm(y ~ 0 + X)), digit = 22)
// print(b, digit = 22)
