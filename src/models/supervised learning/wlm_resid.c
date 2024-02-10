#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
@brief Weighted Linear Regression Residuals
@param coef empty vector to store the output
@param py response vector (example data for model output)
@param w vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void wlm_resid(double *res, double *py, double *w, double *pdta, int *dim) {
    int i, j, k = 0, cnt = 0;
    double itmp, tmp, v;
    double *q, *y, *dta, *iw, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    dta = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    y = (double *) malloc(dim[0] * sizeof(double));
    iw = (double *) malloc(dim[0] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && y && dta && iw) {
        /* Adjust data for the weights */
        for (i = 0; i < dim[0]; i++) {
            iw[cnt] = sqrt(w[i]);
            y[cnt] = py[i] * iw[cnt];
            cnt += (int) (w[i] > 0.0 && isfinite(w[i]));
        }
        for (i = 0; i < dim[0]; i++) {
            #pragma omp for simd
            for (j = 0; j < dim[1]; j++) {
                dta[cnt * j + k] = pdta[*dim * j + i] * iw[k];
            }
            iw[k] = 1.0 / iw[k];
            k += (int) (w[i] > 0.0 && isfinite(w[i]));
        }
        /* Computing matrix Q */
        for (i = 0; i < dim[1]; i++) {
            #pragma omp for simd
            for (j = 0; j < cnt; j++)
                q[*dim * i + j] = dta[*dim * i + j];
            for (k = 0; k < i; k++) {
                tmp = 0.0;
                v = 0.0;
                for (j = 0; j < cnt; j++) {
                    tmp += q[*dim * k + j] * dta[*dim * i + j];
                    v += q[*dim * k + j] * q[*dim * k + j];
                }
                tmp /= v;
                #pragma omp for simd
                for (j = 0; j < cnt; j++) 
                    q[*dim * i + j] -= tmp * q[*dim * k + j];
            }
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < cnt; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            tmp = sqrt(tmp);
            itmp = 1.0 / tmp;
            #pragma omp for simd
            for (j = 0; j < cnt; j++)
                q[*dim * i + j] *= itmp;
        }
        /* Computing least weighted square residuals (Q^t W^{0.5} y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < cnt; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            vec[k] = tmp;
        }
        /* Computing least weighted square residuals y - W^{-0.5} (QQ^t) W^{0.5} y */
        for (j = 0; j < cnt; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = y[j] - iw[j] * tmp;
        }
    }
    free(q);
    free(y);
    free(iw);
    free(vec);
    free(dta);
}

// n <- 400L
// p <- 4L
// X <- matrix(runif(n * p), n, p)
// y <- X %*% rnorm(p, 1) + rnorm(n, 0, .2)
// w <- runif(n, 0.25, 4)
// dyn.load("test.so")
// b <- .C("wlm_coef", res = double(n), y, w, X, dim(X), DUP = FALSE)$res
// print(fit <- lm(y ~ 0 + X, weights = w)$coef, digit = 22)
// print(fabs(resid(fit) - b), digit = 22)
