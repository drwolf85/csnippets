#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DIAG_TOLL 1e-9

#ifdef DEBUG
#include <stdio.h>
#include <time.h>
struct data_vec {
    double *x;
    int n;
    char shog;
};
#endif

/**
 * It computes the outer product of a triangular matrix with itself
 *
 * @param mat the matrix to be transformed
 * @param nn the number of rows and columns in the matrix
 */
void outer_prod_UpperTri(double *mat, int *nn) {
    int i, j, k, n = *nn;
    double tmp;
    for (j = 0; j < n; j++) {
        for (i = 0; i <= j; i++) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * k + j] * mat[n * k + i];
            }
            mat[n * j + i] = tmp;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            mat[n * i + j] = mat[n * j + i];
        }
    }
}

/**
 * It takes the upper triangular part of a square matrix and inverts it
 *
 * @param mat the matrix to be inverted
 * @param nn the number of rows and columns in the matrix
 */
void inverseUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        pos = (n + 1) * (i - 1);
        mat[pos] = mat[pos] != 0.0 ? 1.0 / mat[pos] : 0.0;
        for (j = n - 1; j + 1 > i; j--) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * j + k] * mat[n * k + i - 1];
            }
            mat[n * j + i - 1] = tmp * (- mat[pos]);
        }
    }
}

/**
 * It takes a symmetric matrix and returns the Cholesky decomposition of it
 *
 * @param mat the Hessian matrix
 * @param nn the number of rows and columns in the matrix
 */
void cholHessMat(double *mat, int *nn) {
    int i, j, k = 0;
    double tmp;

    /* Procesing the first row */
    tmp = sqrt(mat[0]);
    mat[k] = tmp;
    tmp = tmp > 0.0 ? 1.0 / tmp : 0.0;
    for (i = 1; i < *nn; i++) mat[*nn * i] *= tmp;
    mat[0] = !isfinite(mat[0]) ? 1.0 : mat[0];
    /* Procesing the other rows */
    for (i = 1; i < *nn; i++) {
        /* Loop for j < i */
        for (j = 0; j < i; j++)
            mat[*nn * j + i] = 0.0;
        /* When j == i */
        k = *nn * i;
        for (j = 0; j < i; j++) {
            tmp = mat[k + j];
            mat[k + i] -= tmp * tmp;
        }
        k += i;
        tmp = sqrt(mat[k]);
        mat[k] = tmp;
        tmp = tmp > 0.0 ? 1.0 / tmp : 0.0;
        /* Loop for j > i */
        for (j = i + 1; j < *nn; j++) {
            for (k = 0; k < i; k++) {
                mat[*nn * j + i] -= mat[*nn * j + k] * mat[*nn * i + k];
            }
            mat[*nn * j + i] *= tmp;
        }
        k = *nn * i + i;
        mat[k] = !isfinite(mat[k]) ? 1.0 : mat[k];
    }
}

/**
 * @brief Inversion of a Hessian Matrix
 *
 * @param mat a (nxn) matrix of real numbers stored by column (column-major format)
 * @param nn the number of rows and columns of the Hessian matrix
 */
void solveHessMat(double *mat, int *nn) {
    cholHessMat(mat, nn); /* Cholesky factorization */
    inverseUT(mat, nn); /* Upper triangular inversion */
    outer_prod_UpperTri(mat, nn); /* Outer product */
}

/**
 * It computes the Noor-Waseem (2009) optimization steps, to minimize a nonlinear error function.
 * The same algorithm can be used to solve a system of nonlinear equations.
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param lambda a pointer to a positive value to stabilize the inversion of the hessian
 * @param grad a routine that computes the gradient of the objective function
 * @param hess a routine that computes the Hessian of the objective function
 */
void noorwaseem(double *param, int *len, int *n_iter, void *info, double *lambda,
                 void (*grad)(double *, double *, int *, void *),
                 void (*hess)(double *, double *, int *, void *)) {
    int t, i, j, np = *len;
    double tmp;
    double *grd_v;
    double *par_v;
    double *hss_m;
    double *hs2_m;

#ifdef DEBUG
    struct data_vec dta = *(struct data_vec *) info;
    dta.shog = 1;
#endif

    grd_v = (double *) malloc(np * sizeof(double));
    par_v = (double *) malloc(np * sizeof(double));
    hss_m = (double *) malloc(np * np * sizeof(double));
    hs2_m = (double *) malloc(np * np * sizeof(double));
    if (grd_v && par_v && hss_m && hs2_m) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            /* Update the Hessian (assumed to be
                    1. Symmetrical! and
                    2. Positive definite!) */
            (*hess)(hss_m, param, len, info);
            for (i = 0; i < np; i++) {
                /* Marquardt's adjustment */
                hss_m[(np + 1) * i] *= 1.0 + *lambda;
                /* Levenberg's adjustment */
                hss_m[(np + 1) * i] += (double) (fabs(hss_m[(np + 1) * i]) < DIAG_TOLL) * DIAG_TOLL;
            }
            /* Invert the Hessian matrix */
            solveHessMat(hss_m, len);
            /* Compute Gauss-Newton descending step */
            #pragma omp parallel for simd private(j, tmp)
            for (i = 0; i < np; i++) {
                tmp = hss_m[np * i] * grd_v[0];
                for (j = 1; j < np; j++) {
                    tmp += hss_m[np * i + j] * grd_v[j];
                }
                par_v[i] = param[i] - tmp;
            }
            /* Add Hessian of the new parameter vector */
            (*hess)(hs2_m, par_v, len, info);
            #pragma omp parallel for private(i, j)
            for (i = 0; i < np; i++) {
                for (j = 0; j < np; j++) {
                    hss_m[np * i + j] += hs2_m[np * i + j];
                }
                /* Compute the mid parameter vector */
                par_v[i] *= 0.5;
                par_v[i] += param[i] * 0.5;
            }
            /* Add four times the Hessian of the mid parameter vector */
            (*hess)(hs2_m, par_v, len, info);
            #pragma omp parallel for private(i, j)
            for (i = 0; i < np; i++) {
                for (j = 0; j < np; j++) {
                    hss_m[np * i + j] += 4.0 * hs2_m[np * i + j];
                    hss_m[np * i + j] *= (1.0 / 6.0);
                }
                /* Marquardt's adjustment */
                hss_m[(np + 1) * i] *= 1.0 + *lambda;
                /* Levenberg's adjustment */
                hss_m[(np + 1) * i] += (double) (fabs(hss_m[(np + 1) * i]) < DIAG_TOLL) * DIAG_TOLL;
            }
            /* Invert new Hessian matrix */
            solveHessMat(hss_m, len);
            /* Compute Noor-Waseem descending step */
            #pragma omp parallel for simd private(j, tmp)
            for (i = 0; i < np; i++) {
                tmp = hss_m[np * i] * grd_v[0];
                for (j = 1; j < np; j++) {
                    tmp += hss_m[np * i + j] * grd_v[j];
                }
                param[i] -= tmp;
            }
        }
#ifdef DEBUG
        (*grad)(grd_v, param, len, (void *) &dta);
#endif
    }
    free(par_v);
    free(grd_v);
    free(hss_m);
    free(hs2_m);
}


#if DEBUG
void my_grad(double *grd, double *par, int *len, void *info) {
    int i;
    double sx = 0.0, sx2 = 0.0, iv = 1.0 / par[1];
    struct data_vec dta = *(struct data_vec *) info;
    if (*len == 2) {
        #pragma omp parallel for simd reduction(+ : sx, sx2)
        for (i = 0; i < dta.n; i++) {
            sx += dta.x[i];
            sx2 += dta.x[i] * dta.x[i];
        }
        grd[0] = par[0] * (double) dta.n - sx;
        grd[1] = (0.5 * (sx2 - 2.0 * par[0] * sx + par[0] * par[0]) * iv - 1.0) * iv;
        if (dta.shog) printf("Grad[0] = %g ; Grad[1] = %g \n", grd[0], grd[1]);
    }
}

void my_hess(double *hss, double *par, int *len, void *info) {
    int i, j;
    double *jcb = (double *) malloc(*len * sizeof(double));
    if (*len == 2 && jcb) {
        my_grad(jcb, par, len, info);
        for (i = 0; i < *len; i++) {
            for (j = 0; j < *len; j++) {
                hss[*len * i + j] = jcb[i] * jcb[j];
            }
        }
    }
    free(jcb);
}

#define N_UNKNOWN 2
#define MAX_ITER 100
#define N_DATA 12 /* This needs to be divisible by the number below */
#define N_BY_LINE 4

int main() {
    double init[N_UNKNOWN] = {0.5 , 1.0 / 12.0};
    int i, len = N_UNKNOWN;
    int maxit = MAX_ITER;
    double lambda = 1e-6;
    struct data_vec v;
    v.x = calloc(N_DATA, sizeof(double));
    v.n = N_DATA;

    printf("Initial values:\n");
    printf("mu = %g and sigma^2 = %g\n", init[0], init[1]);
    if (v.x) {
        /* Init data vector */
        printf("Data values:\n");
        srand(time(NULL));
        for (i = 0; i < N_DATA; i++) {
            v.x[i] = (double) rand() / (double) RAND_MAX;
            printf("%f%s", v.x[i], i % N_BY_LINE == (N_BY_LINE - 1) ? "\n" : " ");
        }
        noorwaseem(init, &len, &maxit, (void *) &v, &lambda, my_grad, my_hess);
        printf("Optimized parameters:\n");
        printf("mu = %g and sigma^2 = %g\n", init[0], init[1]);
    }
    free(v.x);
    return 0;
}
#endif
