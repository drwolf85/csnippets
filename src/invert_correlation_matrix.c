#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * It takes a matrix and returns the outer product of the matrix with itself
 * 
 * @param mat a pointer to the first element of the matrix
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
 * @param nn the dimension of the matrix
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
 * It takes a symmetric matrix and returns the Cholesky decomposition of the matrix
 * 
 * @param mat the matrix to be decomposed
 * @param nn the number of rows and columns in the matrix
 */
void cholCorMat(double *mat, int *nn) {
    int i, j, k;
    double tmp;

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
 * @brief Inversion of a correlation Matrix
 * 
 * @param mat a (nxn) matrix of real numbers stored by column (column-major format)
 * @param nn the number of rows and columns of the correlation matrix
 */
void solveCorMat(double *mat, int *nn) {
    /* Diagonal fix */
    for (int i = 0; i < *nn; i++) mat[(*nn + 1) * i] = 1.0;
    cholCorMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

// n <- 10L
// nn <- 40L
// A <- matrix(runif(n * nn), n, nn)
// S <- pmax(pmin(scale(A), 1), -1)
// F <- t(S) %*% S * (1 / n)
// iA <- F + 0
// diag(iA) <- 1
// print(iA <- solve(iA))
// dyn.load("test.so")
// .C("solveCorMat", F, as.integer(nn), DUP = FALSE)
