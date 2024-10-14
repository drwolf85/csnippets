#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define DISTRO_EPS 1e-12

double dunifsimplex(double *v, int p) {
  int i;
  double z = nan("");
  if (v) {
    z = *v;
    for (i = 1; i < p; i++) z += v[i];
  }
  return (double) (fabs(z - 1.0) < DISTRO_EPS) * gamma((double) p); /* This may not always work */
}

double * runifsimplex(int p) {
  int i;
  unsigned long const m = ~(1 << 31);
  double u;
  double *v = (double *) calloc(p, sizeof(double));
  double z = 0.0;
  if (v) {
    for (i = 0; i < p; i++) {
        u = (0.5 + (double) (rand() & m)) / (1.0 + (double) m);
        v[i] = -log(u);
        z += v[i];
    }
    z = 1.0 / z;
    for (i = 0; i < p; i++) v[i] *= z;
  }
  return v;
}

/* Test function */
#define P 3
#define RUNS 1
int main() {
  int i, j;
  double *x;
  srand(time(NULL));
  printf("Simualtion of a uniform simplex:\n");
  for (j = 0; j < RUNS; j++) {
    x = runifsimplex(P);
    if (x) {      
      for (i = 0; i < P; i++) {
        printf("%f ", x[i]);
      }
      printf("\nDensity: %f\n", dunifsimplex(x, P));
    }
    free(x);
  }
  return 0;
}

