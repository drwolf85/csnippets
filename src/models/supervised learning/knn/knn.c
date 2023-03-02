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

/* Testing functions above */
#ifdef DEBUG
#define N 5
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
    double yref[N] = {0};
    double xref[N][P] = {{0.0,1.2},{1.1,-2.1},{-2.1,2.5},{1.1,-2.1},{2.1,-5.1}};
    for (i = 0; i < N; i++)
        yref[i] = exp(cos(xref[i][0]) * sin(1.0 + xref[i][1])) + M_PI_4;
    printf("True function in (0,0) = %f\n",
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
    return 0;
}
#endif
