#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

/**
 * The function `qrQ` computes the matrix $ of the QR decomposition of a matrix $ using the
 * Gram-Schmidt orthogonalization process
 * 
 * @param dta the data matrix
 * @param dim the dimensions of the matrix
 * @param q the matrix of orthonormal vectors
 */
void qrQ(double *dta, int *dim, double *q) {
    int i, j, k;
    double tmp, v;

    for (i = 0; i < dim[1]; i++) {
        #pragma omp for simd
        for (j = 0; j < dim[0]; j++)
            q[*dim * i + j] = dta[*dim * i + j];
        for (k = 0; k < i; k++) {
            tmp = 0.0;
            v = 0.0;
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * dta[*dim * i + j];
                v += q[*dim * k + j] * q[*dim * k + j];
            }
            tmp /= v;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) 
                q[*dim * i + j] -= tmp * q[*dim * k + j];
        }
        /* Normalization of the column vector */
        tmp = 0.0;
        for (j = 0; j < dim[0]; j++)
            tmp += q[*dim * i + j] * q[*dim * i + j];
        tmp = 1.0 / sqrt(tmp);
        #pragma omp for simd
        for (j = 0; j < dim[0]; j++)
            q[*dim * i + j] *= tmp;
    }
}


// dyn.load("~/Programmi/test_c/qrQ.so")
// a <- matrix(runif(240000), 800, 300)
// system.time(matrix(.C("qrQ", a, dim(a), q =double(240000))$q, 800, 300))
// system.time(qr.Q(qr(a)))
