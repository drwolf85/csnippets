#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifdef DEBUG
#include <stdio.h>
#endif

#define MAX_ITER 100

typedef struct estim {
    double mu;
    double *ma_par;
} estim;

/**
 * @brief Residuals of a linear model (based on OLS)
 * @param res empty vector to store the residuals (in output)
 * @param y response vector (example data for model output)
 * @param dta matrix of data (column-major format)
 * @param dim vector of dimension of `dta` matrix 
 */
void lm_resid(double *res, double *y, double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && vec) {
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
            itmp = 1.0 / sqrt(tmp);
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *= itmp;
        }
        /* Computing least square residuals (Q^t y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            vec[k] = tmp;
        }
        /* Computing least square residuals y - (QQ^t) y */
        for (j = 0; j < dim[0]; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = y[j] - tmp;
        }
    }
    free(q);
    free(vec);
}

void inverseUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        pos = (n + 1) * (i - 1);
        mat[pos] = 1.0 / mat[pos];
        for (j = n - 1; j + 1 > i; j--) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * j + k] * mat[n * k + i - 1];
            }
            mat[n * j + i - 1] = tmp * (- mat[pos]);
        }
    }
}

void lm_coef(double *coef, double *y, double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q,*r;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    r = (double *) calloc(dim[1] * dim[1], sizeof(double));
    memset(coef, 0, dim[1] * sizeof(double));
    if (q && r) {
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
        /* Invert matrix R */
        inverseUT(r, &dim[1]);
        /* Computing regression coefficients */
        for (i = dim[1]; i > 0; i--) {
            k = i - 1;
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            #pragma omp for simd
            for (j = 0; j < i; j++) {
                coef[j] += r[dim[1] * k + j] * tmp;
            }
        }
    }
    free(r);
    free(q);
}

void embed_ma(double *X, double *err, int n, int q) {
    int i, j;
    if (q >= n) return;
    if (X) {
        memset(X, 0, sizeof(double) * n * q);
        for (j = 0; j < q; j++)
            for (i = 0; i < n - j - 1; i++)
                X[n * j + i] = err[j + i + 1];
    }
}

double * target_ma(double *x, int n, int q, double mu) {
    int i;
    double *res;
    if (q >= n) return NULL;
    res = (double *) malloc(n * sizeof(double));
    if (res) for (i = 0; i < n; i++) res[i] = x[i] - mu;
    return res;
}

estim * fit_ma(double *x, int n, int q) {
    int i, j = 0;
    double *y, *X, *err;
    int dim[2];
    estim *res = (estim *) calloc(1, sizeof(estim));
    dim[0] = n;
    dim[1] = q;
    if (x && res) {
        for (i = 0; i < n; i++) res->mu += x[i];
        res->mu /= (double) n;
        X = (double *) calloc(q * n, sizeof(double));
        err = (double *) calloc(n, sizeof(double));
        y = target_ma(x, n, q, res->mu);
        embed_ma(X, y, n, q);
        res->ma_par = (double *) calloc(q, sizeof(double));
        if (err && X && y && res->ma_par) {
            do
            {
                lm_resid(err, y, X, dim);
                embed_ma(X, err, n, q);
                j++;
            } while(j < MAX_ITER);
/*    if (X) {
        printf("Test embed_ma (to be read vertically)\n");
        for (i = 0; i < q * n;) {
            printf("%.2f ", X[i]);
            i++;
            if (i % n == 0) printf("\n");
        }
        printf("\n");
    }*/
            lm_coef(res->ma_par, y, X, dim);
        }
    }
    free(err);
    free(X);
    free(y);
    return res;
}

#ifdef DEBUG
void main() {
    int i;
    int const N = 50;
    int const Q = 3;
    estim *res;
    double x[50] = { 1.04612564, -1.00765414, -0.95323787,  0.89483371, -0.74488022,  0.06825251,
                    -2.50374633, -0.70162665,  0.10846912,  0.86080390,  0.27829369, -1.29050847,
                     0.72689336, -1.22362417, -0.63146275, -3.02374906, -1.13205022,  0.25674405,
                    -0.34068293, -0.73221643,  1.45566137, -0.31540502,  0.14347480, -0.74859040,
                     0.21147942, -1.20158366, -0.60672815, -1.43170568,  0.67910960, -0.24818458,
                     1.02881213, -0.74976112,  0.99874893, -0.35633793, -0.58590449, -0.56258784,
                     0.47390399, -0.27336268, -0.19206190, -0.53284985, -0.87518074,  2.08260175,
                     0.52642028,  0.25721542,  0.59677736,  0.12434588, -0.02864796, -0.75645200,
                     0.02485415,  1.09674453 };
    double *y;
    y = target_ma(x, N, Q, 0.0);
    if (y) {
        printf("Test target_ma\n");
        for (i = 0; i < N; i++) {
            printf("%.2f ", y[i]);
        }
        printf("\n\n");
    }
    if (y) {
        res = fit_ma(x, N, Q);
        printf("Test MA(%d) estimates:\n", Q);
        printf("  mu: %f\n", res->mu);
        for (i = 0; i < Q; i++) printf("  ma%d: %f\n", i, res->ma_par[i]);
    }
    free(res->ma_par);
    free(res);
    free(y);
}
#endif
