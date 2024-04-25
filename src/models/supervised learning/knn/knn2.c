#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct dist_vector {
    double d;
    int i;
} dstvec;

int cmp_dstvec(void const *aa, void const *bb) {
    dstvec a = *(dstvec *)aa;
    dstvec b = *(dstvec *)bb;
    return 2 * ((a.d >= b.d) || (isnan(a.d) || !isnan(b.d))) - 1;
}

/**
 * @brief K-nearest neighbors (using Manhattan distance)
 * 
 * @param x Evaluation point (vector of length `D`)
 * @param dta Pointer to data matrix (stored in row-major format)
 * @param N Number of data points
 * @param D Number of dimensions 
 * @param K Number of neighbors to `x`
 * @return int* 
 */
int * knn1(double *x, double *dta, int N, int D, int K) {
    int i, j, k;
    double dst, tmp;
    int *wh = (int *) malloc(K * sizeof(int));
    dstvec *dsts= (dstvec *) calloc(K, sizeof(dstvec));
    if (wh && dsts) {
        for (k = 0; k < K; k++) {
            dsts[k].i = -1;
            dsts[k].d = INFINITY;
        }
        for (i = 0; i < N; i++) {
            dst = 0.0;
            for (j = 0; j < D; j++) {
                tmp = x[j] - dta[D * i + j];
                dst += fabs(tmp);
            }
            if (dsts[K-1].d > dst) {
                dsts[K-1].d = dst;
                dsts[K-1].i = i;
                qsort(dsts, K, sizeof(dsts), cmp_dstvec);
            }
        }
        for (k = 0; k < K; k++) {
            wh[k] = dsts[k].i;
        }
    }
    free(dsts);
    return wh;
}

/**
 * @brief K-nearest neighbors (using Euclidean distance)
 * 
 * @param x Evaluation point (vector of length `D`)
 * @param dta Pointer to data matrix (stored in row-major format)
 * @param N Number of data points
 * @param D Number of dimensions 
 * @param K Number of neighbors to `x`
 * @return int* 
 */
int * knn2(double *x, double *dta, int N, int D, int K) {
    int i, j, k;
    double dst, tmp;
    int *wh = (int *) malloc(K * sizeof(int));
    dstvec *dsts= (dstvec *) calloc(K, sizeof(dstvec));
    if (wh && dsts) {
        for (k = 0; k < K; k++) {
            dsts[k].i = -1;
            dsts[k].d = INFINITY;
        }
        for (i = 0; i < N; i++) {
            dst = 0.0;
            for (j = 0; j < D; j++) {
                tmp = x[j] - dta[D * i + j];
                dst += tmp * tmp;
            }
            if (dsts[K-1].d > dst) {
                dsts[K-1].d = dst;
                dsts[K-1].i = i;
                qsort(dsts, K, sizeof(dsts), cmp_dstvec);
            }
        }
        for (k = 0; k < K; k++) {
            wh[k] = dsts[k].i;
        }
    }
    free(dsts);
    return wh;
}

/**
 * @brief K-nearest neighbors (using Chebyshev's distance)
 * 
 * @param x Evaluation point (vector of length `D`)
 * @param dta Pointer to data matrix (stored in row-major format)
 * @param N Number of data points
 * @param D Number of dimensions 
 * @param K Number of neighbors to `x`
 * @return int* 
 */
int * knn8(double *x, double *dta, int N, int D, int K) {
    int i, j, k;
    double dst, tmp;
    int *wh = (int *) malloc(K * sizeof(int));
    dstvec *dsts= (dstvec *) calloc(K, sizeof(dstvec));
    if (wh && dsts) {
        for (k = 0; k < K; k++) {
            dsts[k].i = -1;
            dsts[k].d = INFINITY;
        }
        for (i = 0; i < N; i++) {
            dst = 0.0;
            for (j = 0; j < D; j++) {
                tmp = fabs(x[j] - dta[D * i + j]);
                dst += (tmp - dst) * (double) (tmp > dst);
            }
            if (dsts[K-1].d > dst) {
                dsts[K-1].d = dst;
                dsts[K-1].i = i;
                qsort(dsts, K, sizeof(dsts), cmp_dstvec);
            }
        }
        for (k = 0; k < K; k++) {
            wh[k] = dsts[k].i;
        }
    }
    free(dsts);
    return wh;
}

/**
 * @brief K-nearest neighbors (using a generic distance function)
 * 
 * @param x Evaluation point (vector of length `D`)
 * @param dta Pointer to data matrix (stored in row-major format)
 * @param N Number of data points
 * @param D Number of dimensions 
 * @param K Number of neighbors to `x`
 * @return int* 
 */
int * knn_gdf(double *x, double *dta, int N, int D, int K, double (*df_ptr)(double *, double *, int)) {
    int i, j, k;
    double dst, tmp;
    int *wh = (int *) malloc(K * sizeof(int));
    dstvec *dsts= (dstvec *) calloc(K, sizeof(dstvec));
    if (wh && dsts) {
        for (k = 0; k < K; k++) {
            dsts[k].i = -1;
            dsts[k].d = INFINITY;
        }
        for (i = 0; i < N; i++) {
            dst = df_ptr(x, &dta[D * i], D);
            if (dsts[K-1].d > dst) {
                dsts[K-1].d = dst;
                dsts[K-1].i = i;
                qsort(dsts, K, sizeof(dsts), cmp_dstvec);
            }
        }
        for (k = 0; k < K; k++) {
            wh[k] = dsts[k].i;
        }
    }
    free(dsts);
    return wh;
}
