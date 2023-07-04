#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * It computes the outer product of a triangular matrix with itself
 * 
 * @param mat the matrix to be transformed
 * @param nn the number of rows and columns in the matrix
 */
void outer_prod_UpperTri(double *mat, int *nn) {
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
void inverseUT(double *mat, int *nn) {
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
void cholHessMat(double *mat, int *nn) {
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
void solveHessMat(double *mat, int *nn) {
    cholHessMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

/**
 * @brief Compute the inner product of a generic matrix X of size (n x p)
 * 
 * @param res_m a pointer to the symmetric matrix in output
 * @param x a pointer to the input matrix with data (stored as column-major)
 * @param n a pointer to the number of rows of `x`
 * @param p a pointer to the number of columns of `x`
 */
void inner_prod_matrix(double *res_m, double *x, int *n, int *p) {
    int i, j, k;
    double tmp;
    #pragma omp parallel for simd private(j, tmp, k)
    for (i = 0; i < *p; i++) {
        for (j = i; j < *p; j++) {
            tmp = 0.0;
            for (k = 0; k < *n; k++) {
                tmp += x[*n * i + k] * x[*n * j * + k];
            }
            res_m[*p * i + j] = tmp; 
            res_m[*p * j + i] = tmp; 
        }
    }
}

/**
 * It computes the Levenber (1944) optimization steps, to minimize a
 * nonlinear error function using the nonlinear least square approximation. 
 * 
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param lambda penalization parameter (assumed to be greater or equal of zero)
 * @param n_iter number of iterations
 * @param n_obs number of observations
 * @param info a pointer to a structure that contains the data and other information
 * @param error_func a routine to compute a `*n_obs` error vector
 * @param jacobian a routine to compute a `*n_obs`-by-`*len` Jacobian matrix
 */
void levenberg(double *param, int *len, double *lambda,
               int *n_iter, int *n_obs, void *info,
               void (*error_func)(double *, double *, int *, void *),
               void (*jacobian)(double *, double *, int *, void *)) {
    int t, i, j, np = *len;
    int n = *n_obs;
    double tmp;
    double *err_v;
    double *jcb_m;
    double *hss_m;

    err_v = (double *) malloc(n * sizeof(double));
    jcb_m = (double *) malloc(n * np * sizeof(double));
    hss_m = (double *) malloc(np * np * sizeof(double));
    if (err_v && jcb_m && hss_m) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the error vector */
            (*error_func)(err_v, param, len, info);
            /* Update the Jacobian matrix */
            (*jacobian)(jcb_m, param, len, info);
            /* Compute the Hessian */
            inner_prod_matrix(hss_m, jcb_m, n_obs, len);
            for (i = 0; i < np; i++)
                hss_m[(np + 1) * i] += *lambda;
            /* Invert the Hessian matrix */
            solveHessMat(hss_m, len);
            /* Multiply the transposed Jacobian matrix with the error vector */
            #pragma omp parallel for simd private(j, tmp)
            for (i = 0; i < np; i++) {
                tmp = *err_v * jcb_m[n * i];
                for (j = 1; j < n; j++) {
                    tmp += err_v[j] * jcb_m[n * i + j];
                }
                jcb_m[n * i] = tmp;
            }
            /* Compute the descending step */
            #pragma omp parallel for simd private(j, tmp)
            for (i = 0; i < np; i++) {
                tmp = *jcb_m * hss_m[np * i];
                for (j = 1; j < np; j++) {
                    tmp += jcb_m[n * j] * hss_m[np * i + j];
                }
                param[i] -= tmp;
            }
        }
    }
    free(err_v);
    free(jcb_m);
    free(hss_m);
}
