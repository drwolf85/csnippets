#include <stdio.h>
#include <math.h>
#include <omp.h>

double third_moment(double *x, int n) {
  int i;
  double sk = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : sk)
  for (i = 0; i < n; i++) {
   tmp = x[i] * x[i] * x[i];
   sk += tmp;
  }
  sk /= (double) n;
  return sk;
}

double third_central_moment(double *x, int n) {
  int i;
  double mu = 0.0, vr = 0.0, sk = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk)
  for (i = 0; i < n; i++) {
   tmp = x[i];
   mu += tmp;
   tmp *= x[i];
   vr += tmp;
   tmp *= x[i];
   sk += tmp;
  }
  mu /= (double) n;
  vr /= (double) n;
  sk /= (double) n;
  sk -= 3.0 * vr * mu;
  sk += 2.0 * mu * mu * mu;
  return sk;
}

double skewness(double *x, int n) {
  int i;
  double mu = 0.0, sk = 0.0, vr = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk)
  for (i = 0; i < n; i++) {
   tmp = x[i];
   mu += tmp;
   tmp *= x[i];
   vr += tmp;
   tmp *= x[i];
   sk += tmp;
  }
  mu /= (double) n;
  vr /= (double) n;
  sk /= (double) n;
  sk -= 3.0 * mu * vr;
  tmp = mu * mu;
  sk += 2.0 * tmp * mu;
  vr -= tmp;
  vr = sqrt(vr);
  return sk / (vr * vr * vr);
}

#ifdef DEBUG
int main() {
  int N = 7;
  double x[] = {1.0, 2.0, 3.0, 1.1, 1.2, 1.3, 2.4};
  double res = skewness(x, N);
  printf("Third moment of x: %f\n", third_moment(x, N));
  printf("Third central moment of x: %f\n", third_central_moment(x, N));
  printf("Skewness of x: %f\n", res);
  return 0;
}
#endif
