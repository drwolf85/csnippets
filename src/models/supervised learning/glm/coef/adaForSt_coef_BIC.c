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
/**
 * The function `print_mat` prints a matrix of doubles with a specified number of rows and columns.
 * 
 * @param mat This argument is a pointer to a double, which represents a 2D matrix stored in
 * column-major order. The matrix has `nrow` rows and `ncol` columns.
 * @param nrow This argument represents the number of rows in the matrix that you want to print. 
 * It is used to iterate over the rows of the matrix while printing its elements.
 * @param ncol This argument represents the number of columns in the matrix that you want to print.
 * It is used to determine the number of iterations in the inner loop for printing each row of the matrix.
 */
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

/**
 * The function `swap_col` swaps elements between two arrays `a` and `b` of size `n` and calculates the
 * sum of squares of elements in array `a` after swapping.
 * 
 * @param a a pointer to an array of double values representing the first column
 * @param b a pointer to an array of double values used to swap its elements with the corresponding
 * elements in array `a`.
 * @param n The number of elements in the arrays `a` and `b` that are being swapped.
 * 
 * @return The function `swap_col` returns the sum of the squares of the elements in the array `a`
 * after swapping its elements with the elements in the array `b`.
 */
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

/**
 * The function `swap_par` swaps two elements in an array of `param_t` structures.
 * 
 * @param par The pointer to an array of `param_t` type structures
 * @param i An integer representing the index of the first element in the array `par` to swap
 * @param j An integer representing the index of the second element in the array `par` to swap
 */
static inline void swap_par(param_t *par, int i, int j) {
  param_t tmp = par[i];
  par[i] = par[j];
  par[j] = tmp;
}

/**
 * The function `inverseUT` calculates the inverse of an upper triangular matrix in-place.
 * 
 * @param mat A pointer to a matrix assumed to be an upper triangular matrix stored in column-major order.
 * @param nn A pointer to the size of the square matrix being passed to the function.
 */
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

/**
 * The function `ols_coef` calculates ordinary least squares coefficients using matrix operations and
 * parallel processing.
 * 
 * @param par The `par` parameter in the `ols_coef` function represents an array where the computed
 * coefficients will be stored. These coefficients are calculated based on the input parameters `y`,
 * `q`, `r`, `dim`, `mnp`, `cnt`, and the intermediate calculations performed within the function.
 * @param y The parameter `y` in the `ols_coef` function represents an array of double values. It is
 * used to store the values of the dependent variable in a linear regression model.
 * @param q The parameter `q` is a pointer to a double array containing the Q matrix of the Gram-Schmidt decomposition.
 * @param r The parameter `r` is a pointer to a double array representing the R matrix of the Gram-Schmidt decomposition.
 * @param dim An integer array that contains two elements. The first element `dim[0]` represents the number of rows, 
 * and the second element `dim[1]` represents the number of columns of Q.
 * @param mnp The total number of column of matrix Q.
 * @param cnt The effective number of parameters to estimates using submatrices.
 */
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
    for (i = cnt; i > 0; i--) {
      tmp = 0.0;
      for (j = 0; j < dim[0]; j++) {
          tmp += q[dim[0] * (i - 1) + j] * y[j];
      }
      #pragma omp for simd
      for (j = 0; j < i; j++) {
          par[j] += ir[cnt * (i - 1) + j] * tmp;
      }
    }
  }
  free(ir);
}

/**
 * The function `resid` calculates the residual error by subtracting the product of parameters and
 * matrix elements from the input array `y`.
 * 
 * @param err The `err` parameter is a pointer to a double array where the residuals will be stored.
 * @param y The `y` parameter is a pointer to an array of double values.
 * @param dimX The `dimX` parameter is a pointer to an integer array that contains two elements. The
 * first element `dimX[0]` represents the number of elements in the `err` and `y` arrays, while the
 * second element `dimX[1]` represents the total number of variables.
 * @param cnt The variable `cnt` represents the number of non-zero elements in the `par` array. It is
 * used in the loop to iterate over the elements of the `par` array and perform calculations based on
 * its values.
 * @param tmx A pointer to a double array corresponding to the transposed matrix of original covariates.
 * @param par A pointer to a structure type `param_t` with the values and memory locations of estimated parameters.
 */
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

/**
 * The function calculates the Bayes Information Criterion using parallel processing and SIMD-vectorization
 * instructions.
 * 
 * @param err The `err` parameter is an array of double values representing errors.
 * @param dimX The `dimX` parameter is a pointer to an integer array that contains two elements. The
 * first element `dimX[0]` represents the number of elements in the `err` and `y` arrays, while the
 * second element `dimX[1]` represents the total number of variables.
 * @param cnt The count of non-zero parameters or the degree of freedom of the fitted model. It is 
 * used in the calculation of the Bayes Information Criterion (BIC) as a factor that penalizes the 
 * model complexity based on the number of data points.
 * @param step_sz A constant value used in the calculation of the Bayesian Information Criterion (BIC).
 * It is a tuning parameter that influences the penalty term in the BIC formula.
 * 
 * @return The function `BayesInfoCriterion` is returning a double value.
 */
static inline double BayesInfoCriterion(double *err, int *dimX, int cnt, double const step_sz) {
  double res = 0.0;
  int i;
  #pragma omp parallel for simd private(i) reduction(+ : res)
  for (i = 0; i < dimX[0]; i++) {
    res += err[i] * err[i];
  }
  res /= (double) dimX[0];
  // res += 0.5 * step_sz * (double) cnt * log((double) dimX[0]); /* According to Schwarz */
  res = (double) dimX[0] * log(res) + step_sz * (double) cnt * log((double) dimX[0]); /* according to Raymaekers et al. (2023) */
  return res;
}

/**
 * The function `adaForSt` implements an adaptive forward stepwise regression algorithm with BIC for early stopping rule.
 * 
 * @param card_AS The `card_AS` parameter in the function `adaForSt` is a pointer to an integer that
 * will be used to store the number of variables in the active set after the function completes its
 * execution. This value represents the cardinality of the active set of variables selected by the
 * algorithm.
 * @param y The `y` parameter in the function `adaForSt` represents an array of double values that
 * contains the response variable values for the regression problem.
 * @param X The parameter `X` in the provided function `adaForSt` is a pointer to a double array
 * representing the input data matrix. It is used for various calculations within the function, such as
 * computing gradients, residuals, and performing matrix operations like transposition and
 * orthogonalization.
 * @param dimX The `dimX` parameter represents the dimensions of the input matrix `X`. It is an array
 * containing two integers: `dimX[0]` represents the number of rows in `X`, and `dimX[1]` represents
 * the number of columns in `X`.
 * @param M The parameter `M` represents the maximum number of iterations for the AFS algorithm. 
 * It is used to control the number of iterations the algorithm will perform before terminating.
 * @param step_sz The `step_sz` parameter in the provided function `adaForSt` represents the step size
 * used in updating the model coefficients during the AFS algorithm. It controls the amount by which 
 * the model coefficients are adjusted in each iteration of the algorithm.
 * 
 * @return The function `adaForSt` returns a pointer to a `param_t` structure.
 */
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

      /*  BIC for early stopping rule */
      nBIC = BayesInfoCriterion(err, dimX, cnt, step_sz);
    
      /* Sequential loop for Adaptive ForSt */
      for (m = 0; m < (size_t) M && cnt < mnp && nBIC < oBIC; m++) {
        oBIC = nBIC;
        // #ifdef DEBUG
        // printf("BIC: %g\n", oBIC);
        // #endif
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

        /* BIC for early stopping rule */
        nBIC = BayesInfoCriterion(err, dimX, cnt, step_sz);
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
