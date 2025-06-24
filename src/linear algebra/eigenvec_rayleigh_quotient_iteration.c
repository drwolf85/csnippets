#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Rayleigh Quotient Method to Compute the Eigenvector Associated with the Largest Eigenvalues of a Symmetric Positive-Definite Matrix */

#define EPS_TOLL_MAX 1e-8
#define EPS_TOLL_MIN 1e-16

/**
 * It computes the outer product of a triangular matrix with itself
 * 
 * @param mat the matrix to be transformed
 * @param nn the number of rows and columns in the matrix
 */
static inline void outer_prod_UpperTri(double *mat, size_t *nn) {
    size_t i, j, k, n = *nn;
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
static inline void inverseUT(double *mat, size_t *nn) {
    size_t i, j, k, pos, n = *nn;
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
 * @param mat the covariance matrix
 * @param nn the number of rows and columns in the matrix
 */
static inline void cholCovMat(double *mat, size_t *nn) {
    size_t i, j, k = 0;
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
 * @brief Inversion of a covariance Matrix
 * 
 * @param mat a (nxn) matrix of real numbers stored by column (column-major format)
 * @param nn the number of rows and columns of the covariance matrix
 */
static inline void solveCovMat(double *mat, size_t *nn) {
    cholCovMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

/** 
 * The function runif() is a C function that generates a random number from a uniform distribution with
 * lower bound `lb` and upper bound `ub`
 * 
 * @param lb lower bound of the uniform distribution
 * @param ub upper bound of the uniform distribution
 * 
 * @return A random number from a uniform distribution
 */
static inline double runif(double lb, double ub) {
   uint64_t u;
   u = (uint64_t) arc4random() << 32ULL;
   u |= (uint64_t) arc4random();
   return fmin(lb, ub) + ldexp((double) u, -64) * fabs(ub - lb);
}

/**
 * @brief Ratio between quadratic forms (xt A x)/(xt x)
 * 
 * @param x Pointer to a vector approximating an eigenvector
 * @param A Pointer to a symmetric positive-definite matrix (stored in column-major format)
 * @param n Number of rows (or columns) of the matrix `A`
 * 
 * @return double
 */
static inline double ratio_quad_forms(double *x, double *A, size_t n) {
    double num = 0.0, den = 0.0;
    double tmp;
    size_t i, j;
    for (i = 0; i < n; i++) {
        tmp = x[i] * x[i];
        den += tmp;
        num += tmp * A[i * (n + 1)];
        for (j = 0; j < i; j++) {
            num += 2.0 * x[i] * x[j] * A[i * n + j];
        }
    }
    return num / den;
}

/**
 * @brief Compute the maximum of the absolute values in a vector
 * 
 * @param y Pointer to a vector of real number (i.e., proportional approximation to the eigenvector)
 * @param n Number of components in the vector `y`
 * 
 * @return double 
 */
static inline double vmax(double *y, size_t n) {
    size_t i;
    double res = 0.0;
    for (i = 0; i < n; i++) res = fmax(res, fabs(y[i]));
    return res;
}

/**
 * @brief Product between a Matrix and a column vector 
 * 
 * @param y Pointer where to store the output (column) vector
 * @param iA Pointer to an inverted squared symmetric positive-definite matrix
 * @param x Pointer to the (column) vector to multiply
 * @param n Number of components in `x` and `y`
 */
static inline void mat_dot_vec(double *y, double *iA, double *x, size_t n) {
    size_t i, j;
    for (i = 0; i < n; i++) {
        y[i] = iA[n * i] * x[0];
        for (j = 1; j < n; j++) {
            y[i] += iA[n * i + j] * x[j];
        }
    }
}

static inline void sqr_norm(double *x, size_t n) {
    double tmp = 0.0;
    size_t i;
    for (i = 0; i < n; i++) tmp += x[i] * x[i];
    tmp = 1.0 / sqrt(tmp);
    for (i = 0; i < n; i++) x[i] *= tmp;
}

/**
 * @brief Rayleigh Quotient Iteration Return one eigenvector
 * 
 * @param A Pointer to a symmetric positive-definite matrix (stored in column-major format)
 * @param n Number of rows (or columns) of the matrix `A`
 * 
 * @return A pointer to the eigen 
 */
extern double * rayleigh_quotient_iteration(double *A, size_t n) {
    double *x = NULL;
    double *y = NULL;
    double *iA = NULL;
    double sigma, imax, err;
    size_t i, nn;
    if (__builtin_expect(A && n > 0, 1)) {
        nn = n * n;
        x = (double *) malloc(n * sizeof(double));
        y = (double *) malloc(n * sizeof(double));
        iA = (double *) malloc(nn * sizeof(double));
        if (__builtin_expect(x && y && iA, 1)) {
            memcpy(iA, A, sizeof(double) * nn);
            for (i = 0; i < n; i++) x[i] = runif(0.0, 1.0);
            /* Normalized iterative power method */
            do {
                mat_dot_vec(y, iA, x, n);
                imax = 1.0 / vmax(y, n);
                /* Update the vector and compute the convergence rate/error rate */
                sigma = *y * imax;
                err = fabs(*x - sigma);
                *x = sigma;
                for (i = 1; i < n; i++) {
                    sigma = y[i] * imax;
                    err += fabs(x[i] - sigma);
                    x[i] = sigma;
                }
                err /= (double) n;
            } while (err > EPS_TOLL_MAX);
            /* Rayleigh quotient method */
            do {
                sigma = ratio_quad_forms(x, A, n); /* Rayleigh Quotient (a.k.a. the shift) */
                for (i = 0; i < n; i++) iA[i * (n + 1)] = A[i * (n + 1)] - sigma; /* Fixing diagonal of `iA` */
                /* Solve (A - sigma I) y = x for y */
                solveCovMat(iA, &n);
                mat_dot_vec(y, iA, x, n);
                 /* Compute the normalization constant */
                imax = 1.0 / vmax(y, n);
                /* Update the vector and compute the convergence rate/error rate */
                sigma = *y * imax;
                err = fabs(*x - sigma);
                *x = sigma;
                for (i = 1; i < n; i++) {
                    sigma = y[i] * imax;
                    err += fabs(x[i] - sigma);
                    x[i] = sigma;
                }
                err /= (double) n;
                memcpy(iA, A, sizeof(double) * nn); /* Replacing `iA` with `A`*/
            } while(err > EPS_TOLL_MIN);
        }
        sqr_norm(x, n);
        if (__builtin_expect(iA != NULL, 1)) free(iA);
        if (__builtin_expect(y != NULL, 1)) free(y);
    }
    return x;
}

#ifdef DEBUG
int main(void) {
    size_t const N = 3;
    double A[] = { 2.4981235, 1.0050413, 0.1397135, 1.0050413, 2.0813131, -0.2228291, 0.1397135, -0.2228291, 2.0135784 };
    double *x = rayleigh_quotient_iteration(A, N);
    size_t i;
    for (i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
    if (x) free(x);
    return 0;
}
#endif
