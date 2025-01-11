#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct param {
  double v;
  int i;
} param_t;

double swap_col(double *a, double *b, int n) {
  int i;
  double res = 0.0;
  if (a && b && a != b) {
    #pragma omp parallel for reduction(+ : res)
    for (i = 0; i < n; i++) {
      double tmp = a[i];
      a[i] = b[i];
      b[i] = tmp;
      res += a[i] * a[i];
    }
  }
  return res;
}

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

double * ls_coef(double *y, double *q, double *r, int *dim, int mnp, int cnt) {
  int i, j;
  double tmp;
  double *par = (double *) calloc(cnt, sizeof(double));
  double *ir = (double *) calloc(cnt * cnt, sizeof(double));
  if (y && q && r && dim && par && ir) {
    #pragma omp parallel for private(i, j) collapse(2)
    for (j = 0; j < cnt; j++) {
      for (i = 0; i < cnt; i++) {
        ir[cnt * j + i] = r[mnp * j + i];
      }
    }
    inverseUT(ir, &cnt);
    for (j = cnt - 1; j >= 0; j--) {
      tmp = 0.0;
      for (i = 0; i < *dim; i++) {
          tmp += q[*dim * j + i] * y[i];
      }
      #pragma omp for simd
      for (i = 0; i <= j; i++) {
          par[j] += ir[cnt * j + i] * tmp;
      }
    }
  }
  free(ir);
  return par;
}

extern param_t * adaForSt(double *y, double *X, int *dimX, int M, double const step_sz) {
  param_t *par = NULL;
  int *idx;
  int mnp, cnt = 0;
  double *grd = NULL;
  double *err = NULL;
  double *tmx = NULL;
  double *Xcp = NULL;
  double *tpr = NULL;
  size_t m, i, j, k;
  double *q, *r;
  double mxgr, tmp;

  if (y && X && dimX && M > 0 && step_sz > 0.0 && step_sz <= 1.0) {
    mnp = dimX[0] < dimX[1] ? dimX[0] : dimX[1];
    par = (param_t *) calloc(dimX[1], sizeof(param_t)); /* coefficient (and active set) */
    idx = (int *) calloc(dimX[1], sizeof(int)); /* index of coefficients */
    grd = (double *) calloc(dimX[1], sizeof(double)); /* gradient */
    err = (double *) malloc(dimX[0] * sizeof(double)); /* residuals */
    tmx = (double *) malloc(dimX[0] * dimX[1] * sizeof(double)); /* transposition of X */
    Xcp = (double *) malloc(dimX[0] * dimX[1] * sizeof(double)); /* copy of X */
    q = (double *) malloc(dimX[0] * mnp * sizeof(double));
    r = (double *) calloc(mnp * dimX[1], sizeof(double));
    if (idx && par && grd && err && tmx && Xcp && q && r) {
      /* Copy `X' in 'Xcp` for QR updates */
      memcpy(Xcp, X, dimX[0] * dimX[1] * sizeof(double));
      /* Indexing for QR updates */
      #pragma omp parallel for
      for (j = 0; j < dimX[1]; j++) {
        idx[j] = j;
      }

      /* Transpose X for quick data access */
      #pragma omp parallel for private(i, j) collapse(2)
      for (i = 0; i < dimX[0]; i++) {
        for (j = 0; j < dimX[1]; j++) {
          tmx[dimX[1] * i + j] = X[dimX[0] * j + i];
        }
      }
      /*  LASSO... */      
      
      /* Sequential loop for Adaptive ForSt */
      for (m = 0; m < (size_t) M && cnt < mnp /** TODO: include an early stopping rule */; m++) {

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
        for (j = 0; j < dimX[1]; j++) {
          tmp = fabs(grd[j]);
          k += (size_t) (mxgr < tmp) * ((int) j - k);
          mxgr += (double) (mxgr < tmp) * (tmp - mxgr);
        }
        /* Update the active set */
        for (j = 0; j < cnt && par[j].i != k && j < dimX[1]; j++);
        if (j == cnt && j < dimX[1]) {
          par[j].i = k;
          idx[cnt] ^= idx[k];
          idx[k] ^= idx[cnt];
          idx[cnt] ^= idx[k];

          /* Swap data columns and variable indices*/
          tmp = swap_col(&Xcp[dimX[0] * cnt], &Xcp[dimX[0] * k], dimX[0]);
          swap_col(&r[mnp * cnt], &r[mnp * k], mnp);
          tmp = sqrt(tmp);
          r[cnt * mnp + cnt] = tmp; /* Update matrix R */

          /* Update matrix Q */
          tmp = 1.0 / tmp;
          #pragma omp parallel for
          for (i = 0; i < dimX[0]; i++) {
            q[dimX[0] * cnt + i] = Xcp[dimX[0] * cnt + i] * tmp;
          }
          /* Orthogonalization */
          #pragma omp parallel for private(j, i)
          for (j = cnt + 1; j < dimX[1]; j++) {
              for (i = 0; i < dimX[0]; i++)
                  r[j * mnp + cnt] +=  q[dimX[0] * cnt + i] * Xcp[dimX[0] * j + i];
              for (i = 0; i < dimX[0]; i++) 
                  Xcp[dimX[0] * j + i] -= q[dimX[0] * cnt + i] * r[j * mnp + cnt];
          }

          /* Update the count of the active set */
          cnt++;
        }
        /* OLS step based on variables from the active set */
        tpr = ls_coef(y, q, r, dimX, mnp, cnt);
        if (tpr) {/* if a valid pointer, update the parameter estiamtes */
          #pragma omp parallel for simd private(j)
          for (j = 0; j < cnt; j++) {
            par[j].v *= (1.0 - step_sz);
            par[j].v = step_sz * tpr[j];
          }
        }
        free(tpr);
      }
    }
    free(par);
    free(idx);
    free(grd);
    free(err);
    free(tmx);
    free(Xcp);
    free(q);
    free(r);
  }
  return par;
}
