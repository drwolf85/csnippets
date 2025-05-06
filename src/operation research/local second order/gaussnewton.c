#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * It computes the outer product of a triangular matrix with itself
 *
 * @param mat the matrix to be transformed
 * @param nn the number of rows and columns in the matrix
 */
static inline void outer_prod_UpperTri(double *mat, int *nn) {
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
 * It takes the upper triangular part of a square matrix and inverts it
 *
 * @param mat the matrix to be inverted
 * @param nn the number of rows and columns in the matrix
 */
static inline void inverseUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        pos = (n + 1) * (i - 1);
        mat[pos] = mat[pos] != 0.0 ? 1.0 / mat[pos] : 0.0;
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
 * It takes a symmetric matrix and returns the Cholesky decomposition of it
 *
 * @param mat the Hessian matrix
 * @param nn the number of rows and columns in the matrix
 */
static inline void cholHessMat(double *mat, int *nn) {
    int i, j, k = 0;
    double tmp;

    /* Procesing the first row */
    tmp = sqrt(mat[0]);
    mat[k] = tmp;
    tmp = tmp > 0.0 ? 1.0 / tmp : 0.0;
    for (i = 1; i < *nn; i++) mat[*nn * i] *= tmp;
    mat[0] = !isfinite(mat[0]) ? 1.0 : mat[0];
    /* Procesing the other rows */
    for (i = 1; i < *nn; i++) {
        /* Loop for j < i */
        for (j = 0; j < i; j++)
            mat[*nn * j + i] = 0.0;
        /* When j == i */
        k = *nn * i;
        for (j = 0; j < i; j++) {
            tmp = mat[k + j];
            mat[k + i] -= tmp * tmp;
        }
        k += i;
        tmp = sqrt(mat[k]);
        mat[k] = tmp;
        tmp = tmp > 0.0 ? 1.0 / tmp : 0.0;
        /* Loop for j > i */
        for (j = i + 1; j < *nn; j++) {
            for (k = 0; k < i; k++) {
                mat[*nn * j + i] -= mat[*nn * j + k] * mat[*nn * i + k];
            }
            mat[*nn * j + i] *= tmp;
        }
        k = *nn * i + i;
        mat[k] = !isfinite(mat[k]) ? 1.0 : mat[k];
    }
}

/**
 * @brief Inversion of a Hessian Matrix
 *
 * @param mat a (nxn) matrix of real numbers stored by column (column-major format)
 * @param nn the number of rows and columns of the Hessian matrix
 */
static inline void solveHessMat(double *mat, int *nn) {
    cholHessMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

/**
 * It computes the Gauss-Newton optimization steps, to minimize a
 * nonlinear error function using the nonlinear least square approximation.
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param grad a routine that computes the gradient of the objective function
 * @param hess a routine that computes the Hessian of the objective function
 */
extern void gaussnewton(double *param, int *len, int *n_iter, void *info,
                 void (*grad)(double *, double *, int *, void *),
                 void (*hess)(double *, double *, int *, void *)) {
    int t, i, j, np = *len;
    double tmp;
    double *grd_v;
    double *hss_m;

    grd_v = (double *) malloc(np * sizeof(double));
    hss_m = (double *) malloc(np * np * sizeof(double));
    if (grd_v && hss_m) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            /* Update the Hessian (assumed to be 
                    1. Symmetrical! and 
                    2. Positive definite!) */
            (*hess)(hss_m, param, len, info);
            /* Invert the Hessian matrix */
            solveHessMat(hss_m, len);
            /* Compute descending step */
            #pragma omp parallel for simd private(j, tmp)
            for (i = 0; i < np; i++) {
                tmp = hss_m[np * i] * grd_v[0];
                for (j = 1; j < np; j++) {
                    tmp += hss_m[np * i + j] * grd_v[j];
                }
                param[i] -= tmp;
            }
        }
    }
    free(grd_v);
    free(hss_m);
}
