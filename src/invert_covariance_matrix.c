#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

void cholCovMat(double *mat, int *nn) {
    int i, j, k;
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
void solveCovMat(double *mat, int *nn) {
    cholCovMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

// n <- 10L
// nn <- 4L
// A <- matrix(runif(n * nn, -1), n, nn)
// F <- var(A)
// print(iF <- solve(F))
// dyn.load("test.so")
// .C("solveCovMat", F, as.integer(nn), DUP = FALSE)
