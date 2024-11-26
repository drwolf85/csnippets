#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct param {
  double v;
  int i;
} param_t;

static inline void inverseUT(double *mat, int *nn) {
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

static inline double * lm_coef(param_t *coef, double *y, double *dta, int *dim, int cnt) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *r, *p;

    q = (double *) malloc(dim[0] * cnt * sizeof(double));
    r = (double *) calloc(cnt * cnt, sizeof(double));
    p = (double *) calloc(cnt, sizeof(double));
    if (p && q && r) {
        for (i = 0; i < cnt; i++) {
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] = dta[*dim * coef[i].i + j];
            for (k = 0; k < i; k++) {
                tmp = 0.0;
                v = 0.0;
                for (j = 0; j < dim[0]; j++) {
                    tmp += q[*dim * k + j] * dta[*dim * coef[i].i + j];
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
                q[*dim * i + j] *= itmp;
        }
        /* Invert matrix R */
        inverseUT(r, &cnt);
        /* Computing regression coefficients */
        for (i = cnt; i > 0; i--) {
            k = i - 1;
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            #pragma omp for simd
            for (j = 0; j < i; j++) {
                p[j] += r[dim[1] * k + j] * tmp;
            }
        }
    }
    free(r);
    free(q);
    return p;
}

extern param_t * adaForSt(double *y, double *X, int *dimX, int M, double const step_sz) {
  /* Code written assuming `X` matrix is stored in column major format */
  double mxgr, tmp;
  int mnp, cnt = 0;
  param_t *par = NULL;
  double *tpr = NULL;
  double *grd = NULL;
  double *err = NULL;
  double *tmx = NULL;
  size_t m, i, j, k;
  if (y && X && dimX && M > 0 && step_sz > 0.0 && step_sz <= 1.0) {
    par = (param_t *) calloc(dimX[1], sizeof(param_t)); /* coefficient */
    grd = (double *) calloc(dimX[1], sizeof(double)); /* gradient */
    err = (double *) malloc(dimX[0] * sizeof(double)); /* residuals */
    tmx = (double *) malloc(dimX[0] * dimX[1] * sizeof(double)); /* residuals */
    mnp = dimX[0] < dimX[1] ? dimX[0] : dimX[1];
    if (par && err && tmx && grd && tpr) {
      /* Transpose X for quick data access */
      #pragma omp parallel for private(i, j) collapse(2)
      for (i = 0; i < dimX[0]; i++) {
        for (j = 0; j < dimX[1]; j++) {
          tmx[dimX[1] * i + j] = X[dimX[0] * j + i];
        }
      }
      for (m = 0; m < (size_t) M && cnt < mnp; m++) { /* Sequential loop: Do not make it parallel! */
        /* Compute residuals */
        memcpy(err, y, sizeof(double) * dimX[0]);
        #pragma omp parallel for simd private(i, j)
        for (i = 0; i < dimX[0]; i++) {
          for (j = 0; j < cnt; j++) {
            err[i] -= par[j].v * tmx[dimX[1] * i + par[j].i];
          }
        }
        /* Gradient */
        #pragma omp parallel for simd private(j, i)
        for (j = 0; j < dimX[1]; j++) {
          grd[j] = tmx[j] * err[0];
          for (i = 1; i < dimX[0]; i++) {
            grd[j] += X[dimX[0] * j + i] * err[i];
          }
        }
        /* Find the variable with the maximum-absolute-gradient value */
        k = 0;
        mxgr = -0.5;
        #pragma omp parallel for simd private(j, tmp) reduction(+ : mxgr, k)
        for (j = 0; j < dimX[1]; j++) {
          tmp = fabs(grd[j]);
          k += (size_t) (mxgr < tmp) * ((int) j - k);
          mxgr += (double) (mxgr < tmp) * (tmp - mxgr);
        }
        /* Update the active set */
        for (j = 0; j < cnt && par[j].i != k && j < dimX[1]; j++);
        if (j == cnt && j < dimX[1]) {
          par[j].i = k;
          cnt++;
        }
        /* OLS step based on variables from the active set */
        tpr = lm_coef(par, y, X, dimX, cnt);
        if (tpr) { /* if a valid pointer, update the parameter estiamtes */
          #pragma omp parallel for simd private(j)
          for (j = 0; j < cnt; j++) {
            par[j].v *= (1.0 - step_sz);
            par[j].v = step_sz * tpr[j];
          }        
        }
        free(tpr);
      }
    }
    free(grd);
    free(tmx);
    free(err);
  }
  return par;
}

