#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void qrR(double *dta, int *dim, double *r) {
    int i, j, k;
    double itmp, tmp, v;
    double *q;
    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    memset(r, 0, dim[1] * dim[1] * sizeof(double));
    if(q) for (i = 0; i < dim[1]; i++) {
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
            r[dim[1] * i + k] = tmp / sqrt(v);
            tmp /= v;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) 
                q[*dim * i + j] -= tmp * q[*dim * k + j];
        }
        /* Normalization of the column vector */
        tmp = 0.0;
        for (j = 0; j < dim[0]; j++)
            tmp += q[*dim * i + j] * q[*dim * i + j];
        tmp = sqrt(tmp);
        r[dim[1] * i + i] = tmp;
        itmp = 1.0 / tmp;
        #pragma omp for simd
        for (j = 0; j < dim[0]; j++)
            q[*dim * i + j] *=itmp;
    }
    free(q);
}

// dyn.load("~/Programmi/test_c/qrR.so")
// a <- matrix(runif(24), 8, 3)
// system.time(matrix(.C("qrR", a, dim(a), r = double(9))$r, 3, 3))
// system.time(qr.R(qr(a)))