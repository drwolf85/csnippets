#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
 * The function `qrQR_mgs` (Modified Gram-Schmidt) takes
 * a matrix `dta` of dimension `dim[0]` by `dim[1]`
 * and returns the QR decomposition of `dta` in `q` and `r`
 *
 * @param dta the data matrix
 * @param dim the dimensions of the matrix
 * @param q the q matrix
 * @param r the upper triangular matrix
 */
void qrQR_mgs(double *dta, int *dim, double *q, double *r) {
    int i, j, k;
    double itmp, tmp, v;

    memset(r, 0, dim[1] * dim[1] * sizeof(double));
    memcpy(q, dta, dim[0] * dim[1] * sizeof(double));
    for (i = 0; i < dim[1]; i++) {
        /* Normalization of the column vector */
        tmp = 0.0;
        for (j = 0; j < dim[0]; j++)
            tmp += q[*dim * i + j] * q[*dim * i + j];
        tmp = sqrt(tmp);
        r[dim[1] * i + i] = tmp;
        itmp = 1.0 / tmp;
        #pragma omp for simd
        for (j = 0; j < dim[0]; j++)
            q[*dim * i + j] *= itmp;
        /* Orthogonalization */
        for (j = i + 1; j < dim[1]; j++) {
            for (k = 0; k < dim[0]; k++)
                r[dim[1] * j + i] += q[*dim * i + k] * q[*dim * j + k];
            for (k = 0; k < dim[0]; k++)
                q[*dim * j + k] -= q[*dim * i + k] * r[dim[1] * j + i];
        }
    }
}

// dyn.load("~/Programmi/test_c/qrQR.so")
// a <- matrix(runif(24), 8, 3)
// system.time(res <- .C("qrQR_mgs", a, dim(a), q = double(24), r = double(9))[c("q", "r")])
// array(res$q, dim=dim(a))
// matrix(res$r, ncol(a), ncol(a))
// array(res$q, dim=dim(a))%*%matrix(res$r, ncol(a), ncol(a))
// system.time(qr.R(qr(a)))
