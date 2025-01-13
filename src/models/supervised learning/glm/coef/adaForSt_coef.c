#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct param {
  double v;
  int i;
} param_t;

#ifdef DEBUG
void print_mat(double *mat, int nrow, int ncol) {
  int i, j;
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%.6f ", mat[nrow * j + i]);
    }
    printf("\n");
  }
}
#endif

static inline double swap_col(double *a, double *b, int n) {
  int i;
  double res = 0.0;
  if (a && b) {
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

static inline void swap_par(param_t *par, int i, int j) {
  param_t tmp = par[i];
  par[i] = par[j];
  par[j] = tmp;
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

static inline void ols_coef(double *par, double *y, double *q, double *r, int *dim, int mnp, int cnt) {
  int i, j;
  double tmp;
  double *ir = (double *) calloc(cnt * cnt, sizeof(double));
  if (par && y && q && r && dim && par && ir) {
    /* Set parameters to zero */
    memset(par, 0, dim[1] * sizeof(double));
    /* Copy submatrix R */
    #pragma omp parallel for private(i, j) collapse(2)
    for (j = 0; j < cnt; j++) {
      for (i = 0; i < cnt; i++) {
        ir[cnt * j + i] = r[mnp * j + i];
      }
    }
    /* Invert the submatrix R */
    inverseUT(ir, &cnt);
    /*#ifdef DEBUG
      printf("Inverse submatrix R:\n");
      print_mat(ir, cnt, cnt);
      printf("\n");
    #endif*/
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
}

static inline void resid(double *err, double *y, int *dimX, int cnt, double *tmx, param_t *par) {
  int i, j;
  #pragma omp parallel for simd private(i, j)
  for (i = 0; i < dimX[0]; i++) {
    err[i] = y[i];
    for (j = 0; j < cnt; j++) {
      err[i] -= par[j].v * tmx[dimX[1] * i + par[j].i];
    }
  }
}
static inline double BayesInfoCriterion(double *err, int *dimX, int cnt) {
  double res = 0.0;
  int i;
  #pragma omp parallel for simd private(i) reduction(+ : res)
  for (i = 0; i < dimX[0]; i++) {
    res += err[i] * err[i];
  }
  res /= (double) dimX[0];
  // res -= 0.5 * (double) cnt * log((double) dimX[0]);
  res = (double) dimX[0] * log(res) - (double) cnt * log((double) dimX[0]); /* according to Raymaekers et al. (2023) */
  return res;
}

extern param_t * adaForSt(int *card_AS, double *y, double *X, int *dimX, size_t M, double const step_sz) {
  param_t *par = NULL;
  int mnp, cnt = 0;
  double *grd = NULL;
  double *err = NULL;
  double *tmx = NULL;
  double *Xcp = NULL;
  double *tpr = NULL;
  size_t m, i, j, k;
  double *q, *r;
  double mxgr, tmp;
  double oBIC = INFINITY;
  double nBIC = 0.0;

  if (y && X && dimX && M > 0 && step_sz > 0.0 && step_sz <= 1.0) {
    mnp = dimX[0] < dimX[1] ? dimX[0] : dimX[1];
    if (mnp < 1) return par;
    par = (param_t *) calloc(dimX[1], sizeof(param_t)); /* coefficient (and active set) */
    grd = (double *) malloc(dimX[1] * sizeof(double)); /* gradient */
    tpr = (double *) malloc(dimX[1] * sizeof(double)); /* OLS parameters */
    err = (double *) malloc(dimX[0] * sizeof(double)); /* residuals */
    tmx = (double *) malloc(dimX[0] * dimX[1] * sizeof(double)); /* transposition of X */
    q = (double *) malloc(dimX[0] * dimX[1] * sizeof(double));
    r = (double *) calloc(mnp * dimX[1], sizeof(double));
    if (par && tpr && grd && err && tmx && q && r) {
      /* Initialize the vector of parameters */
      #pragma omp parallel for simd
      for (j = 0; j < dimX[1]; j++) {
        par[j].i = j;
      }
      /* Copy `X' in 'q` for QR updates */
      memcpy(q, X, dimX[0] * dimX[1] * sizeof(double));

      /* Transpose X for quick data access */
      #pragma omp parallel for private(i, j) collapse(2)
      for (i = 0; i < dimX[0]; i++) {
        for (j = 0; j < dimX[1]; j++) {
          tmx[dimX[1] * i + j] = X[dimX[0] * j + i];
        }
      }

      /* Compute residuals */
      resid(err, y, dimX, cnt, tmx, par);

      /*  LASSO... or BIC for early stopping rule */
      nBIC = BayesInfoCriterion(err, dimX, cnt);
    
      /* Sequential loop for Adaptive ForSt */
      for (m = 0; m < (size_t) M && cnt < mnp && nBIC < oBIC; m++) {
        oBIC = nBIC;
        /* Gradient */
        #pragma omp parallel for private(j, i)
        for (j = 0; j < dimX[1]; j++) {
          grd[par[j].i] = X[dimX[0] * par[j].i] * err[0];
          for (i = 1; i < dimX[0]; i++) {
            grd[par[j].i] += X[dimX[0] * par[j].i + i] * err[i];
          }
        }

        /* Find the variable with the maximum-absolute-gradient value */
        k = 0;
        mxgr = -0.5;
        for (j = 0; j < dimX[1]; j++) {
          tmp = fabs(grd[par[j].i]);
          k += (size_t) (tmp > mxgr) * ((int) j - k);
          mxgr += (double) (tmp > mxgr) * (tmp - mxgr);
        }
        /* Update the active set */
        for (j = 0; j < cnt && par[j].i != k && j < dimX[1]; j++);
        if (j == cnt && j < dimX[1] && k >= cnt) {
          /*#ifdef DEBUG
          printf("%d <swap> %d\n", j, k);
          #endif*/
          /* Swap variable indices and data columns */
          swap_par(par, j, k);
          swap_col(&r[mnp * j], &r[mnp * k], mnp);
          tmp = swap_col(&q[dimX[0] * j], &q[dimX[0] * k], dimX[0]);
          tmp = sqrt(tmp);
          r[cnt * (mnp + 1)] = tmp; /* Update matrix R */
          
          /* #ifdef DEBUG
            printf("\nMatrix R:\n");
            print_mat(r, mnp, mnp);
          #endif */

          /* Update matrix Q */
          tmp = 1.0 / tmp;
          #pragma omp parallel for
          for (i = 0; i < dimX[0]; i++) {
            q[dimX[0] * cnt + i] *= tmp;
          }
          /* Orthogonalization */
          #pragma omp parallel for private(j, i)
          for (j = cnt + 1; j < dimX[1]; j++) {
            for (i = 0; i < dimX[0]; i++)
                r[j * mnp + cnt] += q[dimX[0] * cnt + i] * q[dimX[0] * j + i];
            for (i = 0; i < dimX[0]; i++) 
                q[dimX[0] * j + i] -= q[dimX[0] * cnt + i] * r[mnp * j + cnt];
          }

          /* Update the count of the active set */
          cnt++;

          /*#ifdef DEBUG
            printf("\nMatrix Q:\n");
            print_mat(q, dimX[0], cnt);
            printf("\n");
          #endif*/

          /* OLS step based on variables from the active set */
          ols_coef(tpr, y, q, r, dimX, mnp, cnt);
        }

        /* Update the model coefficients*/
        #pragma omp parallel for simd private(j)
        for (j = 0; j < cnt; j++) {
          par[j].v *= (1.0 - step_sz);
          par[j].v += step_sz * tpr[j];
        }

        /* Compute residuals */ 
        resid(err, y, dimX, cnt, tmx, par);

        /*  LASSO... or BIC for early stopping rule */
        nBIC = BayesInfoCriterion(err, dimX, cnt);
      }
    }
    #ifdef DEBUG
    printf("Residuals:\n");
    print_mat(err, 1, dimX[0]);
    printf("Y values:\n");
    print_mat(y, 1, dimX[0]);
    printf("\n");
    #endif
    free(tpr);
    free(grd);
    free(err);
    free(tmx);
    free(q);
    free(r);
    *card_AS = cnt;
    #ifdef DEBUG
    printf("The algorithm stopped at m = %d\n", m);
    #endif
  }
  return par;
}

#ifdef DEBUG
#define MY_M_ITERATION_VALUE 500
#define MY_STEP_SISE_VALUE 0.05
int main() {
  double x_mat[]  = {-0.626873, -0.283008, -0.045327, -0.852087, -0.763049, 0.062491, 0.545433, -0.441383, 0.720966, 0.3389, 0.847938, -0.496264, -0.987421, -1.398132, -1.669215, -0.033717, 0.863076, 0.077906, -1.785289, -1.044279, -1.148154, -0.38202, 1.171428, -0.470829, -0.694094, 1.05329, 1.68112, -0.459551, 0.492357, 0.222412, 0.003783, 0.654647, 1.050526, 1.577019, -0.6913, -0.39524, -1.25327, -2.410438, 0.699714, 0.830185, 0.626838, 0.395043, 1.521474, -0.096075, -0.456023, -0.440329, 1.602557, -0.426096, 0.63762, -1.057338, -0.344068, -2.013045, -0.914089, -0.612044, -0.24745, -0.163447, -0.376161, 0.645006, -0.337815, -0.999168};
  int x_dim[] = {6, 10}; /* True active set has cardinality three */
  double y[] = {0.403424, -0.515348, -2.074205, 2.094899, 2.617164, 0.246223, 0.0, 0.0, 0.0, 0.0};
  int active_set = 0;
  param_t *beta_hat = adaForSt(&active_set, y, x_mat, x_dim, MY_M_ITERATION_VALUE, MY_STEP_SISE_VALUE);
  if (beta_hat) {
    printf("Adaptive Selection Forward Regression (M = %d, ada_step = %g):\n", MY_M_ITERATION_VALUE, MY_STEP_SISE_VALUE);
    for (int i = 0; i < active_set && i < 10; i++) {
      printf("(estim %d of variab %d) %g\n", i, beta_hat[i].i, beta_hat[i].v);
    }
    free(beta_hat);
    /* Approximately, Variab 0) -2.75692  Variab 1) -0.52521  Variab 2) 1.18201 */
  }
  return 0;
}
#endif
