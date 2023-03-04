/**
 * The function `inverseUT` takes a square matrix `mat` and its dimension `n` as input, and returns
 * the inverse of the upper triangular matrix `mat` in place
 * 
 * @param mat the matrix to be inverted
 * @param nn the dimension of the matrix
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

// n <- 4L
// A <- F <- matrix(runif(n * n), n, n)
// A[lower.tri(A)] <- 0
// F[lower.tri(F)] <- 0
// print(iA <- solve(A))
// dyn.load("test.so")
// .C("inverseUT", F, as.integer(n), DUP = FALSE)
