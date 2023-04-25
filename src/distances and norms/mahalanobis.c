#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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
 * @param mat the covariance matrix
 * @param nn the number of rows and columns in the matrix
 */
void cholCovMat(double *mat, int *nn) {
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
 * The function `sqrtmUT` takes a square matrix `mat` and its dimension `n` as input, and returns
 * the matrix-valued square root of the upper triangular matrix `mat` in place
 * 
 * @param mat the matrix to compute the square root
 * @param nn the dimension of the matrix
 */
void sqrtmUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp, res;
    for (i = 0; i < n; i++) {
        pos = (n + 1) * i;
        mat[pos] = sqrt(mat[pos]);
        for (j = i - 1; j >= 0; j--) {
            tmp = 0.0;
            for (k = j + 1; k < i; k++) {
                tmp += mat[n * i + k] * mat[n * k + j];
            }
            res = mat[pos] + mat[(n + 1) * j];
            mat[n * i + j] -= tmp;
            mat[n * i + j] /= res;
        }
    }
}

/**
 * @brief Inversion of the square root of a covariance Matrix
 * 
 * @param mat a (nxn) matrix of real numbers stored by column (column-major format)
 * @param nn the number of rows and columns of the covariance matrix
 */
void invSqrtCovMat(double *mat, int *nn) {
    cholCovMat(mat, nn); /* Cholesky factorization */
    sqrtmUT(mat, nn); /* Square root of the matrix */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

/**
 * The function calculates the Mahalanobis distance between two vectors using a given covariance
 * matrix.
 * 
 * @param x A pointer to an array of doubles representing the first vector.
 * @param y The parameter `y` is a pointer to a double array representing the second vector for which
 * we want to compute the Mahalanobis distance from the first vector `x`.
 * @param n The number of dimensions in the vectors x and y.
 * @param sigma The parameter `sigma` is a pointer to an array of size `n*n` containing the covariance
 * matrix or rotation matrix used to compute the Mahalanobis distance.
 * 
 * @return a double value, which represents the Mahalanobis distance between two vectors x and y of
 * size n, using the covariance matrix sigma.
 */
double mahalanobis_distance(double *x, double *y, size_t n, double *sigma) {
    double res = 0.0;
    double *z, *zz, *Hm;
    size_t i, j;
    int nn = (int) n;
    
    z = (double *) malloc(n * sizeof(double));
    zz = (double *) calloc(n, sizeof(double));
    Hm = (double *) malloc(n * n * sizeof(double));

    if (z && zz && Hm) {
        /* Prepare the covariance/rotation matrix */
        #pragma omp parallel for simd 
        for (i = 0; i < n * n; i++) Hm[i] = sigma[i];
        invSqrtCovMat(Hm, &nn);
        /* Compute the difference between the two vectors */
        #pragma omp parallel for simd 
        for (i = 0; i < n; i++) z[i] = x[i] - y[i];
        #pragma omp parallel for simd private(j) 
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                zz[i] += z[j] * Hm[n * i + j];
            }
        }
        #pragma omp parallel for simd reduction(+ : res)
        for (i = 0; i < n; i++) {
            res += zz[i] * zz[i];
        }
    }
    free(z);
    free(zz);
    free(Hm);
    return sqrt(res);
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2};
    double y[] = {1.2, 1.2};
    double S[] = {1.23, -0.5, -0.5, 3.21};
    size_t i;
    printf("Mahalanobis distance between x and y is %f\n", mahalanobis_distance(x, y, 2, S));
    return 0;
}
