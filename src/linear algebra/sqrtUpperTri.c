#include <math.h>

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

// library(expm)
// n <- 4L
// A <- F <- matrix(runif(n * n), n, n)
// A[lower.tri(A)] <- 0
// F[lower.tri(F)] <- 0
// print(sqrtm(A))
// dyn.load("test.so")
// .C("sqrtmUT", F, as.integer(n), DUP = FALSE)
