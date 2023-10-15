#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_ITER 10

/**
 * The function `inverseUT` takes a square matrix `mat` and its dimension `n` as input, and returns the
 * inverse of the upper triangular matrix `mat` in place
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
 * It computes the outer product of a matrix with itself.
 * 
 * @param mat the matrix to be transformed
 * @param nn the number of rows and columns in the matrix
 */
void outer_prod_UT(double *mat, int *nn) {
    int i, j, k, n = *nn;
    double tmp;
    for (j = 0; j < n; j++) {
        for (i = 0; i <= j; i++) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * k + j] * mat[n * k + i];
            }
            mat[n * j + i] = tmp;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            mat[n * i + j] = mat[n * j + i];
        }
    }
}

/**
 * The function `lm_coef` computes the regression coefficients of a linear model using the QR
 * decomposition
 * 
 * @param coef the coefficients of the regression
 * @param y the response variable
 * @param dta the data matrix, with each row being a sample and each column being a feature.
 * @param dim a vector of length 2, where dim[0] is the number of observations and dim[1] is the number
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
                r[dim[1] * i + k] = tmp / sqrt(v);
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
            r[dim[1] * i + i] = tmp;
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
}

/**
@brief Logistic (linear) Regression
@param coef empty vector to store the output
@param py response vector (example data for model output)
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
void logis_coef(double *coef, double *py, double *pdta, int *dim) {
    int i, j, k, count = 0;
    double itmp, tmp, v, err = INFINITY, orr = 0.0;
    double *q, *r, *y, *w, *dta, *jcb;
    double const nc = 1.0 / dim[0];

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    r = (double *) calloc(dim[1] * dim[1], sizeof(double));
    dta = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    jcb = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    y = (double *) malloc(dim[0] * sizeof(double));
    w = (double *) malloc(dim[0] * sizeof(double));

    /* memset(coef, 0, dim[1] * sizeof(double)); */
    if (q && r && y && w && dta && jcb) {
        /* Find initial guess via OLS */
        for (i = 0; i < dim[0]; i++) {
            y[i] = 4.0 * py[i] - 2.0;
        }
        lm_coef(coef, y, pdta, dim);
        do {
            /* Computing Jacobian */
            for (j = 0; j < dim[0]; j++) {
                y[j] = 0.0;
                for (i = 0; i < dim[1]; i++) {
                    y[j] += coef[i] * pdta[dim[0] * i + j];
                }
                w[j] = 1.0 / (1.0 + exp(-y[j]));
                y[j] = py[j] - w[j];
                w[j] = sqrt(w[j] * (1.0 - w[j]));
                for (i = 0; i < dim[1]; i++)
                    dta[dim[0] * i + j] = w[j] * pdta[dim[0] * i + j];
            }
            /* Computing matrix Q (and R) of diag(w) %*% X */
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
                /* Normalization */
                tmp = 0.0;
                for (j = 0; j < dim[0]; j++)
                    tmp += q[*dim * i + j] * q[*dim * i + j];
                tmp = sqrt(tmp);
                r[dim[1] * i + i] = tmp; /* Matrix R */
            }
            /* Invert matrix R */
            inverseUT(r, &dim[1]);
            outer_prod_UT(r, &dim[1]);
            /* Computing regression coefficients */
            #pragma omp for simd private(tmp, j)
            for (i = 0; i < dim[1]; i++) {
                tmp = 0.0;
                for (j = 0; j < dim[0]; j++) {
                    tmp += y[j] * pdta[dim[0] * i + j];
                }
                q[i] = tmp;
            }
            #pragma omp for simd private(tmp, j)
            for (i = 0; i < dim[1]; i++) {
                tmp = 0.0;
                for (j = 0; j < dim[1]; j++) {
                    tmp += r[dim[1] * i + j] * q[j];
                }
                coef[i] += tmp;
            }
            count++;
        } while (count < MAX_ITER);
    }
    free(r);
    free(q);
    free(y);
    free(w);
    free(dta);
    free(jcb);
}

// set.seed(0)
// n <- 200000L
// p <- 25L
// X <- matrix(rnorm(n * p), n, p)
// y <- as.double((X %*% rnorm(p, 1) + rnorm(n, 0, 1)) > 0)
// dyn.load("test.so")
// system.time(b <- .C("logis_coef", coef = double(p), y, X, dim(X), DUP = FALSE)$coef)
// system.time(print(cc <- glm(y ~ 0 + X, family = "binomial")$coef, digit = 22))
// print(b, digit = 22)
// sum((y-plogis(X%*%cc))^2); sum((y-plogis(X%*%b))^2)
