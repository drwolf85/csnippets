#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    double *dsts= (double *) calloc(K, sizeof(double));
    for (k = 0; k < N; k++) {
        wh[k] = -1;
        dsts[k] = INFINITY;
    }
    for (i = 0; i < N; i++) {
        dst = 0.0;
        for (j = 0; j < D; j++) {
            tmp = x[j] - dta[D * i + j];
            dst += fabs(tmp);
        }
        for (k = 1; k < K; k++) {
            wh[K - k] += (wh[K - k] - wh[K - k - 1]) * (int) (*dsts > dst);
            dsts[K - k] += (dsts[K - k - 1] - dsts[K - k]) * (double) (*dsts > dst);
        }
        *wh += (i - *wh) * (int) (*dsts > dst);
        *dsts += (dst - *dsts) * (double) (*dsts > dst);
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
    double *dsts= (double *) calloc(K, sizeof(double));
    for (k = 0; k < N; k++) {
        wh[k] = -1;
        dsts[k] = INFINITY;
    }
    for (i = 0; i < N; i++) {
        dst = 0.0;
        for (j = 0; j < D; j++) {
            tmp = x[j] - dta[D * i + j];
            dst += tmp * tmp;
        }
        for (k = 1; k < K; k++) {
            wh[K - k] += (wh[K - k] - wh[K - k - 1]) * (int) (*dsts > dst);
            dsts[K - k] += (dsts[K - k - 1] - dsts[K - k]) * (double) (*dsts > dst);
        }
        *wh += (i - *wh) * (int) (*dsts > dst);
        *dsts += (dst - *dsts) * (double) (*dsts > dst);
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
    double *dsts= (double *) calloc(K, sizeof(double));
    for (k = 0; k < N; k++) {
        wh[k] = -1;
        dsts[k] = INFINITY;
    }
    for (i = 0; i < N; i++) {
        dst = 0.0;
        for (j = 0; j < D; j++) {
            tmp = fabs(x[j] - dta[D * i + j]);
            dst += (tmp - dst) * (double) (tmp > dst);
        }
        for (k = 1; k < K; k++) {
            wh[K - k] += (wh[K - k] - wh[K - k - 1]) * (int) (*dsts > dst);
            dsts[K - k] += (dsts[K - k - 1] - dsts[K - k]) * (double) (*dsts > dst);
        }
        *wh += (i - *wh) * (int) (*dsts > dst);
        *dsts += (dst - *dsts) * (double) (*dsts > dst);
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
    double *dsts= (double *) calloc(K, sizeof(double));
    for (k = 0; k < N; k++) {
        wh[k] = -1;
        dsts[k] = INFINITY;
    }
    for (i = 0; i < N; i++) {
        dst = df_ptr(x, &dta[D * i], D);
        j = (int) (*dsts > dst);
        for (k = 1; k < K; k++) {
            wh[K - k] += (wh[K - k] - wh[K - k - 1]) * j;
            dsts[K - k] += (dsts[K - k - 1] - dsts[K - k]) * (double) j;
        }
        *wh += (i - *wh) * j;
        *dsts += (dst - *dsts) * (double) j;
    }
    free(dsts);
    return wh;
}
