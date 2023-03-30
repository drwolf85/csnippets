/**
 * It computes the inner product of a matrix with itself.
 * 
 * @param mat a pointer to the matrix
 * @param nn the number of rows/columns in the matrix
 */
void inner_prod_UpperTri(double *mat, int *nn) {
    int i, j, k, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        mat[i - 1] = mat[n * (i - 1)] * mat[0];
        for (j = 1; j <= i; j++) {
            tmp = 0.0;
            for (k = 0; k <= j; k++) {
                tmp += mat[n * (i - 1) + k] * mat[n * j + k];
            }
            mat[n * j + i - 1] = tmp;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            mat[n * j + i] = mat[n * i + j];
        }
    }
}

// n <- 4L
// A <- F <- matrix(runif(n * n), n, n)
// A[lower.tri(A)] <- 0
// R <- t(A) %*% A
// dyn.load("test.so")
// matrix(.C("inner_prod_UpperTri", A, as.integer(n), DUP = FALSE)[[1L]], n, n)
