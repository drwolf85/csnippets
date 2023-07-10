#include <omp.h>

/**
 * @brief Compute the outer product of a generic matrix X of size (n x p)
 * 
 * @param res_m a pointer to the symmetric matrix (n x n) in output
 * @param x a pointer to the input matrix with data (stored as column-major)
 * @param n a pointer to the number of rows of `x`
 * @param p a pointer to the number of columns of `x`
 */
void outer_prod_matrix(double *res_m, double *x, int *n, int *p) {
    int i, j, k;
    double tmp;
    #pragma omp parallel for simd private(j, tmp, k)
    for (i = 0; i < *n; i++) {
        for (j = i; j < *n; j++) {
            tmp = 0.0;
            for (k = 0; k < *p; k++) {
                tmp += x[*n * i + k] * x[*n * j * + k];
            }
            res_m[*n * i + j] = tmp; 
            res_m[*n * j + i] = tmp; 
        }
    }
}
