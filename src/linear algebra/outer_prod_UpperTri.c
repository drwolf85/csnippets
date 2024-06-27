/**
 * It takes a matrix and returns the outer product of the matrix with itself
 * 
 * @param mat a pointer to the first element of the matrix
 * @param nn the number of rows and columns of the matrix
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

// n <- 4L
// A <- F <- matrix(runif(n * n), n, n)
// A[lower.tri(A)] <- 0
// R <- A %*% t(A)
// dyn.load("test.so")
// matrix(.C("outer_prod_UpperTri", A, as.integer(n), DUP = FALSE)[[1L]], n, n)
