#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct projection_vector {
    double y;
    double x;
    double w;
    uint32_t i;
} prjvec;

typedef struct vector_st {
    double v;
    double w;
    size_t i;
} vector;

static inline double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -30) - 1.0;
   s = a * a;
   b = ldexp((double) v, -30) - 1.0;
   s += b * b * (1.0 - s);
   s = -2.0 * log(s) / s;
   a = b * sqrtf(s);
   return mu + sd * a;
}

int cmp_prj_knn(void const *aa, void const *bb) {
    prjvec a = *(prjvec *) aa;
    prjvec b = *(prjvec *) bb;
    return 2 * (a.x >= b.x) - 1;
}

int cmp_vector(void const *aa, void const *bb) {
    vector a = *(vector *) aa;
    vector b = *(vector *) bb;
    return 2 * (int) ((a.v >= b.v) || (isnan(a.v) && !isnan(b.v))) - 1;
}

int cmp_double(void const *aa, void const *bb) {
    double a = *(double *) aa;
    double b = *(double *) bb;
    return 2 * (int) ((a >= b) || (isnan(a) && !isnan(b))) - 1;
}

double wt_med_std(vector *a, size_t n) {
    size_t i;
    double res = nan("");
    double sum = 0.0;
    double cw;
    
    if (a) {
        for (i = 0; i < n; i++) sum += a[i].w;
        sum *= 0.5;
        qsort(a, n, sizeof(vector), cmp_vector);
        res = a[0].v;
        cw = a[0].w;
        for (i = 1; i < n && cw < sum; i++) {
            res = a[i].v;
            cw += a[i].w;
        }
    }
    return res;
}

/**
 * @brief Robust self-smoothing matrix
 * 
 * @param X Pointer to a matrix
 * @param dimX Pointer to number of rows and columns
 * @param w Pointer to a vector of weights
 * @param nj Pointer to the number of random projections to perform
 * @param nk Pointer ot the number of nearest neighbors to use
 */
void robust_self_smoothing_matrix(double *X, int *dimX, double *w, int *nj, int *nk) {
    uint32_t i, j, k, s;
    uint32_t n = (uint32_t) *dimX;
    uint32_t p = (uint32_t) dimX[1];
    uint32_t knnsz = (uint32_t) *nk;
    uint32_t K = (uint32_t) *nj;
    uint32_t J = p >> 1;
    int64_t mnp, mxp;
    double isrp, rnd;
    double *finX = (double *) malloc(n * p * sizeof(double)); 
    double *res = (double *) malloc(n * K * sizeof(double));
    double *rtz = (double *) malloc(p * sizeof(double));
    prjvec *prj = (prjvec *) malloc(n * sizeof(prjvec));
    vector *a = (vector *) calloc(n, sizeof(vector));
    isrp = 1.0 / sqrt((double) J);
    if (n > 0 && p > 1 && K > 0 && prj && a && rtz && res && finX) {
        srand(time(NULL));
        for (s = 0 ; s < p; s++) {
            for (k = 0; k < K; k++) {
                /* Initialize the rotation vector */
                memset(rtz, 0, p * sizeof(double));
                for (j = 0; j < J; j++) rtz[j] = rnorm(0.0, isrp);
                for (j = 0; j < p; j++) {
                    i = (uint32_t) rand() % p;
                    rnd = rtz[j];
                    rtz[j] = rtz[i];
                    rtz[i] = rnd;
                }
                rtz[s] = 0.0;
                /* Compute the random projection */
                memset(prj, 0, sizeof(prjvec) * n);
                for (i = 0; i < n; i++) {
                    prj[i].x += X[j * n + i] * rtz[0]; /** FIXME: Clamp X here */
                    prj[i].y = X[s * n + i];
                    prj[i].w = w == NULL ? 1.0 : w[i];
                    prj[i].i = i;
                }
                for (j = 1; j < p; j++) {
                    for (i = 0; i < n; i++) {
                        prj[i].x += X[j * n + i] * rtz[j]; /** FIXME: Clamp X here */
                    }
                }
                /* Find nearest neighbors in the projection */
                qsort(prj, n, sizeof(prjvec), cmp_prj_knn);
                /* Compute the approximated knn weighted medians */
                for (i = 0; i < n; i++) {
                    mnp = (int64_t) i - (int64_t) (knnsz >> 1);
                    mxp = (int64_t) i + (int64_t) (knnsz >> 1);
                    mnp *= (mnp > 0);
                    mxp += (mxp > (int64_t) n) * ((int64_t) n - mxp);
                    for (j = mnp; j < mxp; j++) {
                        a[j].v = prj[j].y;
                        a[j].w = prj[j].w;
                    }
                    res[k + K * i] = wt_med_std(&a[mnp], (size_t) (mxp - mnp));
                }
            }
            /* Compute the median of medians */
            k = K >> 1;
            for (i = 0; i < n; i++) {
                j = K * i;
                qsort(&res[j], K, sizeof(double), cmp_double);
                res[j] = res[j + k] * 0.5 + 0.5 * res[j + k - (size_t)!(K & 1)];
            }
            finX[n * s] = res[0];
            for (i = 1; i < n; i++) finX[n * s + i] = res[K * i];
        }
        for (i = 0; i < n * p; i++) {
            X[i] = finX[i];
        }
    }
    free(a);
    free(prj);
    free(rtz);
    free(res);
    free(finX);
}

/*
dyn.load("test.so")
data(iris)
dta <- scale(log(iris[, 1:4]))
w <- double(nrow(dta)) + 1
res <- .C("robust_self_smoothing_matrix", res = as.matrix(dta), dim(dta), w, maxit = 10L, knn = 5L, DUP = FALSE)$res
res <- array(res, dim = dim(dta))
pairs(res, col = as.integer(as.factor(iris$Species)))
dta <- iris[, 1:4]
w <- double(nrow(dta)) + 1
res <- .C("robust_self_smoothing_matrix", res = as.matrix(dta), dim(dta), w, maxit = 10L, knn = 5L, DUP = FALSE)$res
res <- array(res, dim = dim(dta))
pairs(res, col = as.integer(as.factor(iris$Species)))
*/

#ifdef DEBUG
#include "../../.data/iris.h"
int main() {
    int j, i = 0, maxit = 1000, knn = 5L;
    double w[N], dta[N * P] = {0};
    int dim[2] = {N, P};
    for (; i < N; i++) w[i] = 1.0;
    for (i = 0; i < N * P; i++) dta[i] = x_iris[i];
    robust_self_smoothing_matrix(dta, dim, w, &maxit, &knn);
    for (i = 0; i < N; i++) { /*
        for (j = 0; j < P; j++)
            printf("%1.2f ", dta[i + N * j]);
        printf("\t");
        for (j = 0; j < P; j++)
            printf("%1.2f ", x_iris[i + N * j]);*/
        for (j = 0; j < P; j++)
            printf("%s%1.2f ", x_iris[i + N * j] >= dta[i + N * j] ? " " : "", x_iris[i + N * j] - dta[i + N * j]);            
        printf("\n"); 
    }
    return 0;
}
#endif

