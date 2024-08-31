#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MY_EPS_DIV 1e-12
#define MY_EPS_NORM 1e-14

/* Operations on a single vector */

void v_zeros(double *v, size_t n) {
    memset(v, 0, sizeof(double) * n);
}

void v_ones(double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        v[i] = 1.0;
    }
}

void v_neg(double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        v[i] = -v[i];
    }
}

void v_inv(double *v, size_t n, bool by_zero) {
    size_t i;
    if (by_zero) {
        for (i = 0; i < n; i++) {
            v[i] = 1.0 / v[i];
        }
    } 
    else {
        for (i = 0; i < n; i++) {
            v[i] = fabs(v[i]) < MY_EPS_DIV ? 1.0 : 1.0 / v[i];
        }
    }
}

void v_apply(double *v, size_t n, double (*f)(double)) {
    for (size_t i = 0; i < n; i++) {
        v[i] = f(v[i]);
    }
}

double v_norm(double *x, size_t n) {
    double res = 0.0;
    for (size_t i = 0; i < n; i++) res += x[i] * x[i];
    return sqrt(res);
}

void v_normalize(double *x, size_t n) {
    double res = v_norm(x, n);
    if (res > MY_EPS_NORM) {
        res = 1.0 / res;
        for (size_t i = 0; i < n; i++) x[i] *= res;
    }
}

double v_sum(double *x, size_t n) {
    double res = 0.0;
    for (size_t i = 0; i < n; i++) res += x[i];
    return res;
}

double v_mean(double *x, size_t n) {
    double res = v_sum(x, n);
    if (n > 0) res /= (double) n;
    return res;
}

/* Scalar-vector operations */

void sv_add(double *a, double b, size_t n) {
for (size_t i = 0; i < n; i++) {
        a[i] += b;
    }
}

void sv_sub(double *a, double b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] -= b;
    }
}

void sv_mul(double *a, double b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= b;
    }
}

void vs_div(double *a, double b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] /= b;
    }
}

void sv_div(double *a, double b, size_t n, bool by_zero) {
    size_t i;
    if (by_zero){
        for (i = 0; i < n; i++) {
            a[i] = b / a[i];
        }
    }
    else {
        for (i = 0; i < n; i++) {
            a[i] = fabs(a[i]) < MY_EPS_DIV ? 1.0: b / a[i];
        }
    }
}

void vs_apply(double *a, double b, size_t n, 
              double (*f)(double, double)) {
    size_t i;
    for (i = 0; i < n; i++) {
        a[i] = f(a[i], b);
    }
}

void sv_apply(double a, double *b, size_t n, 
              double (*f)(double, double)) {
    for (size_t i = 0; i < n; i++) {
        b[i] = f(a, b[i]);
    }
}

/* Vector-vector operations */

void vv_add(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void vv_sub(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] -= b[i];
    }
}

void vv_mul(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void vv_div(double *a, double *b, size_t n, bool by_zero) {
    size_t i;
    if (by_zero){
        for (i = 0; i < n; i++) {
            a[i] /= b[i];
        }
    }
    else {
        for (i = 0; i < n; i++) {
            a[i] /= fabs(b[i]) < MY_EPS_DIV ? 1.0 : b[i];
        }
    }
}

void vv_apply(double *a, double *b, size_t n, 
              double (*f)(double, double)) {
    for (size_t i = 0; i < n; i++) {
        a[i] = f(a[i], b[i]);
    }
}

double vv_dot(double *a, double *b, size_t n) {
    double res = 0.0;
    for (size_t i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    return res;
}

double ** vv_cross_dpt(double *a, double *b, size_t n) {
    double **res = NULL;
    bool err_alloc = false;
    size_t i, j;
    res = (double **) malloc(sizeof(double *) * n);
    if (res) {
        for (i = 0; i < n; i++) {
            res[i] = (double *) malloc(sizeof(double *) * n);
            if (res[i]) {
                for (j = 0; j < n; j++) {
                    res[i][j] = a[i] * b[j];
                }
            }
            else {
                err_alloc = true;
            }
        }
        if (err_alloc) {
            for (i = 0; i < n; i++) free(res[i]);
            free(res);
        }
    }
    return res;
}

double * vv_cross_spt(double *a, double *b, size_t n, bool col_maj) {
    double *res = NULL;
    size_t i, j;
    res = (double *) malloc(sizeof(double) * n * n);
    if (res) {
        if (col_maj) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    res[n * j + i] = a[i] * b[j];
                }
            }
        }
        else {
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    res[n * i + j] = a[i] * b[j];
                }
            }
        }
    }
    return res;
}

double vv_dist(double *a, double *b, size_t n) {
    double res = 0.0;
    double tmp;
    for (size_t i = 0; i < n; i++) {
        tmp = a[i] - b[i];
        res += tmp * tmp;
    }
    return sqrt(res);
}
