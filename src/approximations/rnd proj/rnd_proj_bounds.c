#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int lb_lower_dim(double eps, int n) {
  double m = 8.0 / (eps * eps);
  m *= log((double) n);
  return (int) ceil(m);
}

double lb_eps(int m, int n) {
  double eps = 8.0 / (double) m;
  eps *= log((double) n);
  return sqrt(eps);
}

double * error_bounds(double eps, double *v1, double *v2, int d) {
  /*johnson_lindenstrauss_lemma*/
  double *bnd = malloc(2 * sizeof(double));
  double tmp, nrm2 = 0.0;
  int i;
  if (bnd) {
    #pragma omp parallel for simd private(i, tmp) reduction(+ : nrm2)
    for (i = 0; i < d; i++) {
      tmp = v1[i] - v2[i];
      nrm2 += tmp * tmp;
    }
    bnd[0] = (1.0 - eps) * nrm2;
    bnd[0] *= (double) (bnd[0] > 0.0);
    bnd[1] = (1.0 + eps) * nrm2;
  }
  return bnd;
}

double * variance_bounds(double eps, double *Y, int d, int n) {
  /*johnson_lindenstrauss_lemma*/
  double *bnd = malloc(2 * sizeof(double));  
  double tmp, nrm2 = 0.0;
  int i, j, k;
  if (bnd) {
    #pragma omp parallel for simd private(i, j, k, tmp) reduction(+ : nrm2) collapse(2) 
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        if (i > j) {
          for (k = 0; k < d; k++) {
            tmp = Y[i * d + k] - Y[j * d + k];
            nrm2 += tmp * tmp;
          }
        }
      }
    }
    nrm2 /= (double) n * (double) n;
    bnd[0] = (1.0 - eps) * nrm2;
    bnd[0] *= (double) (bnd[0] > 0.0);
    bnd[1] = (1.0 + eps) * nrm2;
  }
  return bnd;
}

#ifdef DEBUG
#define EPS 0.5
#define N 1000
#define D 5
#define M 3

#include "../../.data/data_5x1000_row_maj.h"

int main() {
  double *bounds;
  double my_eps;
  int my_dim;
  
  my_dim = lb_lower_dim(EPS, N);
  printf("My dim = %d, with eps = %f and n = %d\n", my_dim, EPS, N);
  my_eps = lb_eps(M, N);
  printf("My eps = %f, with dimension = %d and n = %d\n", my_eps, M, N);

  bounds = error_bounds(my_eps, x, &x[D], D);
  printf("Error bounds: %f and %f\n", bounds[0], bounds[1]);
  free(bounds);
  bounds = variance_bounds(my_eps, x, D, N);
  printf("Variance bounds: %f and %f\n", bounds[0], bounds[1]);
  free(bounds);
  return 0;
}
#endif

