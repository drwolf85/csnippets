#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    double v;
    size_t i;
}
dst_vec;

int cmp_double(void const *aa, void const *bb) {
    double a = *(double *) aa;
    double b = *(double *) bb;
    if (isnan(a)) return 1;
    if (isnan(b)) return -1;
    return (int) (a >= b) * 2 - 1;
}

double knn_reg(double *x, size_t k, size_t n, size_t p, 
               double **x_ref, double *y_ref, 
               double (*dist)(double *, double *, size_t)) {
    size_t i;
    double res = 0.0;
    dst_vec *dst = (dst_vec *) calloc(n, sizeof(dst_vec));
    if (dst && k > 0 && n > 0 && p > 0) {
        /* Populate the vector of distances */
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            dst[i].v = dist(x, (double *) ((size_t) x_ref + \
                            p * i * sizeof(double *)), p);
            dst[i].i = i;
        }
        /* Sort the distances */
        qsort(dst, n, sizeof(dst_vec), cmp_double);
        /* Compute the average of y */
        #pragma omp parallel for reduction(+ : res)
        for (i = 0; i < k; i++) {
            res += y_ref[dst[i].i];
        }
        res /= (double) k;
    }
    free(dst);
    return res;
}

double knn_idwt_reg(double *x, double alpha, size_t k, 
                    size_t n, size_t p, double **x_ref, double *y_ref, 
                    double (*dist)(double *, double *, size_t)) {
    size_t i;
    double res = 0.0, sm = 0.0, w;
    dst_vec *dst = (dst_vec *) calloc(n, sizeof(dst_vec));
    if (dst && k > 0 && n > 0 && p > 0 && alpha >= 0.0) {
        /* Populate the vector of distances */
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            dst[i].v = dist(x, (double *) ((size_t) x_ref + \
                            p * i * sizeof(double *)), p);
            dst[i].i = i;
        }
        /* Sort the distances */
        qsort(dst, n, sizeof(dst_vec), cmp_double);
        /* Compute the weighted average of y */
        #pragma omp parallel for private(w) reduction(+ : res, sm)
        for (i = 0; i < k; i++) {
            w = 2.0 / (1.0 + pow(fabs(dst[i].v), alpha));
            res += y_ref[dst[i].i] * w;
            sm += w;
        }
        res /= sm;
    }
    free(dst);
    return res;
}

size_t knn_class(double *x, size_t k, size_t n, size_t p, 
               double **x_ref, size_t *y_ref, size_t nc,
               double (*dist)(double *, double *, size_t)) {
    size_t i, j;
    size_t res;
    size_t counts[nc];
    dst_vec *dst = (dst_vec *) calloc(n, sizeof(dst_vec));
    if (dst && k > 0 && n > 0 && p > 0 && nc > 0) {
        /* Initialize the count vector */
        #pragma omp parallel for simd
        for (i = 0; i < nc; i++) counts[i] = 0;
        /* Populate the vector of distances */
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            dst[i].v = dist(x, (double *) ((size_t) x_ref + \
                            p * i * sizeof(double *)), p);
            dst[i].i = i;
        }
        /* Sort the distances */
        qsort(dst, n, sizeof(dst_vec), cmp_double);
        /* Compute the average of y */
        #pragma omp parallel for
        for (i = 0; i < k; i++) {
            #pragma omp atomic update
            counts[y_ref[dst[i].i]] += 1;
        }
        res = 0;
        j = counts[0];
        for (i = 1; i < nc; i++) {
            res += (counts[i] > j) * (i - res);
            j += (counts[i] > j) * (counts[i] - j);
        }
    }
    free(dst);
    return res;
}

size_t knn_idwt_class(double *x, double alpha, size_t k, 
               size_t n, size_t p, double **x_ref, 
               size_t *y_ref, size_t nc,
               double (*dist)(double *, double *, size_t)) {
    size_t i;
    size_t res;
    double w, counts[nc];
    dst_vec *dst = (dst_vec *) calloc(n, sizeof(dst_vec));
    if (dst && k > 0 && n > 0 && p > 0 && nc > 0) {
        /* Initialize the count vector */
        #pragma omp parallel for simd
        for (i = 0; i < nc; i++) counts[i] = 0.0;
        /* Populate the vector of distances */
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            dst[i].v = dist(x, (double *) ((size_t) x_ref + \
                            p * i * sizeof(double *)), p);
            dst[i].i = i;
        }
        /* Sort the distances */
        qsort(dst, n, sizeof(dst_vec), cmp_double);
        /* Compute the average of y */
        #pragma omp parallel for private(w)
        for (i = 0; i < k; i++) {
            w = 1.0 / (1.0 + pow(fabs(dst[i].v), alpha));
            #pragma omp atomic update
            counts[y_ref[dst[i].i]] += w;
        }
        res = 0;
        w = counts[0];
        for (i = 1; i < nc; i++) {
            res += (counts[i] > w) * (i - res);
            w += (counts[i] > w) * (counts[i] - w);
        }
    }
    free(dst);
    return res;
}

#ifdef DEBUG
/* Testing functions above */
#define N 6
#define P 2
double my_dist(double *a, double *b, size_t p) {
    double res = 0.0;
    double tmp;
    for (size_t i = 0; i < p; i++) {
        tmp = a[i] - b[i];
        res += tmp * tmp;
    }
    return sqrt(fabs(res));
}
int main(void) {
    size_t i;
    double alpha;
    double x0[P] = {0};
    double xc[P] = {2.0,-3.0};
    double yref[N] = {0};
    size_t ycls[N] = {1, 0, 1, 0, 1, 2};
    double xref[N][P] = {{0.7,1.3}, {0.0,1.2},{1.1,-1.1},{-2.1,2.5},{0.1,-2.1},{2.1,-5.1}};
    for (i = 0; i < N; i++)
        yref[i] = exp(cos(xref[i][0]) * sin(1.0 + xref[i][1])) + M_PI_4;
    printf("REGRESSION:\nTrue function in (0,0) = %f\n",
           exp(cos(x0[0]) * sin(1.0 + x0[1])) + M_PI_4);
    printf("KNN prediction in (0,0) = %f\n", 
           knn_reg(x0, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 0.0;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 0.5;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 1.0;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 2.0;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 4.0;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    alpha = 8.0;
    printf("IDWT-KNN prediction with alpha = %.1f in (0,0) = %f\n", alpha,
           knn_idwt_reg(x0, alpha, 3, N, P, (double **) xref, yref, my_dist));
    printf("---\nCLASSIFICATION:\nKNN class in (0,0) is %lu\n", 
           knn_class(x0, 2, N, P, (double **) xref, ycls, 3, my_dist));
    alpha = 0.0;
    printf("IDWT-KNN class with alpha = %.1f in (%.1f,%.1f) is %lu\n", alpha, xc[0], xc[1],
           knn_idwt_class(xc, alpha, 2, N, P, (double **) xref, ycls, 3, my_dist));
    alpha = 0.5;
    printf("IDWT-KNN class with alpha = %.1f in (%.1f,%.1f) is %lu\n", alpha, xc[0], xc[1],
           knn_idwt_class(xc, alpha, 2, N, P, (double **) xref, ycls, 3, my_dist));
    alpha = 1.0;
    printf("IDWT-KNN class with alpha = %.1f in (%.1f,%.1f) is %lu\n", alpha, xc[0], xc[1],
           knn_idwt_class(xc, alpha, 2, N, P, (double **) xref, ycls, 3, my_dist));
    alpha = 2.0;
    printf("IDWT-KNN class with alpha = %.1f in (%.1f,%.1f) is %lu\n", alpha, xc[0], xc[1],
           knn_idwt_class(xc, alpha, 2, N, P, (double **) xref, ycls, 3, my_dist));
    return 0;
}
#endif
