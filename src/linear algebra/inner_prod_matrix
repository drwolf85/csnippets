#include <omp.h>

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
