/* Solver of linear equations */
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief L Pointer to a Lower triangular matrix (of size n x n and stored as a column major format)
 * @brief U Pointer to an Uppere triangular matrix (of size n x n and stored as a column major format)
 * @brief perm Pointer to a vector of `n` permuted indices (values between `0` and `n-1`)
 * @brief b Pointer to a vector of `n` real values
 * @brief n Number of values in `b`
 */
double * LUPsolve(double *L, double *U, size_t *perm, double *b, size_t n) {
  double *x = (double *) malloc(n * sizeof(double));
  double *y = (double *) malloc(n * sizeof(double));
  size_t i, j;
  if (__builtin_expect(L && U && perm && b && x && y, 1)) {
    for (i = 0; i < n; i++) {
      y[i] = b[perm[i]];
      for (j = 0; j < i; j++) y[i] -= L[n * j + i] * y[j];
    }
    if (i > 0) {
      i--;
      if (i >= 1) {
        x[i] = y[i] / U[(n + 1) * i];
        for (; i > 0; i--) {
          x[i] = y[i];
          for (j = i + 1; j < n; j++) x[i] -= U[n * j + i] * x[j];
          x[i] /= U[(n + 1) * i];
        }
      }
      *x = *y;
      for (j = 1; j < n; j++) *x -= U[n * j] * x[j];
      *x /= *U;
    }
  }
  if (__builtin_expect(y != NULL, 1)) free(y);
  return x;
}

#ifdef DEBUG

int main(void) {
  double L[] = {1.0, 0.2, 0.6, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0};
  double U[] = {5.0, 0.0, 0.0, 6.0, 0.8, 0.0, 3.0, -0.6, 2.5};
  size_t p[] = {2, 0, 1};
  size_t n = 3ULL;
  double b[] = {3.0, 7.0, 8.0};
  double *x = LUPsolve(L, U, p, b, n);
  size_t i;
  printf("Solution:\n");
  for (i = 0; i < n; i++) printf("%g ", x[i]);
  printf("\n");
  if (x) free(x);
  return 0;
}

#endif
