#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

typedef struct kmp {
    size_t k;
    size_t p;
    size_t *n;
    double *m; 
} kmp;

kmp * param_alloc(size_t k, size_t p) {
    kmp *par = (kmp *) malloc(sizeof(kmp));
    if (par) {
        par->k = k;
        par->p = p;
        par->n = (size_t *) calloc(k, sizeof(size_t));
        par->m = (double *) calloc(k * p, sizeof(double));
        if (!(par->m && par->n)) {
            free(par->n);
            free(par->m);
            free(par);
        }
    }
    return par;
}

void param_free(kmp *par) {
    if (par) {
        free(par->n);
        free(par->m);
    }
    free(par);
}

bool param_check(kmp *par, size_t k, size_t p) {
    bool res = false;
    if (par == NULL) { /* Allocate memory for parameters */
        par = param_alloc(k, p);
        if (par) res = true;
    }
    else { /* Check and fix parameter memory */
        if (par->p != p || par->k != k) {
            free(par);
            par = param_alloc(k, p);
            if (par) res = true;
        } 
        else {
            res = true;
        }
    }
    return res;
}

void set_means(kmp *res, double *data, size_t n) {
    size_t i, j;
    double const in = 1.0 / (double) n;
    #pragma omp parallel for simd private(i)
    for (j = 0; j < res->p; j++) {
        res->m[j] = 0.0;
        for (i = 0; i < n; i++) {
            res->m[j] += data[n * j + i];
        }
        res->m[j] *= in;
    }
    res->n[0] = n;
}

void update_means(kmp *res, double *data, size_t n) {
    if (res && data) {
        if (res->n[0] == 0) {
            set_means(res, data, n);
        }
        else {
            size_t i, j;
            double const in = 1.0 / (double) (n + res->n[0]);
            #pragma omp parallel for simd private(i)
            for (j = 0; j < res->p; j++) {
                res->m[j] *= (double) res->n[0];
                for (i = 0; i < n; i++) {
                    res->m[j] += data[n * j + i];
                }
                res->m[j] *= in;
            }
        }
    }
}

size_t which_group(kmp *par, double *data, size_t n, size_t i) {
    double dst, tmp, mnd = 1.0;
    size_t j, v, res = 0;
    for (j = 0; j < par->k; j++) {
        dst = 0.0;
        for (v = 0; v < par->p; v++) {
            tmp = data[n * v + i] - par->m[par->p * j + v];
            dst += tmp * tmp;
        }
        mnd = (double) (j == 0) * dst + (double) (j > 0) * mnd;
        res = (size_t) (mnd > dst) * j + (size_t) (mnd <= dst) * res;
        mnd = (double) (mnd > dst) * dst + (double) (mnd <= dst) * mnd;
    }
    return res;
}

void update_mwi(kmp *par, double *data, size_t n) {
    size_t i, j, g = par->k * par->p;
    double tmp;
    double *new_m = (double *) calloc(g, sizeof(double));
    if (par && data && new_m) {
        if (par->m && par->n) {
            memset(par->n, 0, sizeof(size_t) * par->k);
            #pragma omp parallel for private(i, g, j)
            for (i = 0; i < n; i++) {
                g = which_group(par, data, n, i);
                for (j = 0; j < par->p; j++) {
                    #pragma omp atomic update
                    new_m[par->p * g + j] += data[n * j + i];
                }
                #pragma omp atomic update
                par->n[g] += 1;
            }
            #pragma omp parallel for private(i, g, j, tmp)
            for (g = 0; g < par->k; g++) {
                if (par->n[g]) {
                    tmp = 1.0 / (double) par->n[g];
                    for (j = 0; j < par->p; j++) {
                        i = par->p * g + j;
                        par->m[i] = new_m[i] * tmp;
                    }
                }
            }
        }
    }
    free(new_m);
}

/**
 * @brief K-means algorithm (for data that are uncorrelated within each group)
 * 
 * @param par Pointer to a parametric structure
 * @param data Pointer to data values (stored as col-major matrix of doubles)
 * @param n Number of data points
 * @param p Number of variables per data point
 * @param k Number of groups
 * @param max_iter Maximum number of iteration to update the k-means
 */
void k_means(kmp *par, double *data, size_t n, size_t p, size_t k, size_t max_iter) {
    size_t h, i, j;
    if (n == 0 || p == 0 || n >= RAND_MAX || n < k) k = 0;
    switch (k) { /* Check if number of group make sense */
        case 0:
            if (par) free(par);
            return;
        case 1:
            param_check(par, 1, p);
            update_means(par, data, n);
            return;
        default:
            if (param_check(par, k, p)) { /* Initialize parameters */
                if (par) if (par->n && par->m) for (h = 0; h < k; h++) { 
                    /** FIXME: It currently pick a point for each group at random 
                               There are other initialization criteria...
                        TODO: Generalize the initialization! */
                    i = (size_t) rand() % n;
                    #pragma omp for simd
                    for (j = 0; j < p; j++) {
                        par->m[p * h + j] = data[n * j + i];
                    }
                }
                /** WARNING: The following loop MUST BE sequential, 
                             DO NOT make it parallel! */
                for (i = 0; i < max_iter; i++) update_mwi(par, data, n);
            }
            break;
    }
}

#ifdef DEBUG
/* Test function */
int main(void) {
    size_t i, j;
    #include "../.data/iris.h"
    kmp *my_km = (kmp *) param_alloc(3, P);
    srand(2023);
    k_means(my_km, x_iris, N, P, 3, 10);
    for (i = 0; i < 3; i++) {
        printf("Group %lu: ", i);
        for (j = 0; j < P; j++) {
            printf("%.5f ", my_km->m[P * i + j]);
        }
        printf("\n");
    }
    param_free(my_km);
    return 0;
}

/* Equivalent code to test `int main(void)` in R:
> kmeans(iris[, -5], iris[c(80, 20, 105), -5], iter.max=10L)
*/
#endif
