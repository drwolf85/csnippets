#include <stdio.h>
#include <math.h>
#include <omp.h>

extern inline double fourth_moment(double *x, int n) {
  int i;
  double kr = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : kr)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i])) {
      tmp = x[i];
      tmp *= x[i];
      tmp *= x[i];
      tmp *= x[i];
      kr += tmp;
    }
  }
  kr /= (double) n;
  return kr;
}

extern inline double fourth_central_moment(double *x, int n) {
  int i;
  double mu = 0.0, vr = 0.0, sk = 0.0, kr = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, kr)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i])) {
      tmp = x[i];
      mu += tmp;
      tmp *= x[i];
      vr += tmp;
      tmp *= x[i];
      sk += tmp;
      tmp *= x[i];
      kr += tmp;
    }
  }
  mu /= (double) n;
  vr /= (double) n;
  sk /= (double) n;
  kr /= (double) n;
  kr -= 4.0 * mu * sk;
  tmp = mu * mu;
  kr += 6.0 * tmp * vr;
  kr -= 3.0 * tmp * tmp;
  return kr;
}

extern inline double kurtosis(double *x, int n) {
  int i;
  double mu = 0.0, sk = 0.0, vr = 0.0, kr = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, kr)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i])) {
      tmp = x[i];
      mu += tmp;
      tmp *= x[i];
      vr += tmp;
      tmp *= x[i];
      sk += tmp;
      tmp *= x[i];
      kr += tmp;
    }
  }
  mu /= (double) n;
  vr /= (double) n;
  sk /= (double) n;
  kr /= (double) n;
  kr -= 4.0 * mu * sk;
  tmp = mu * mu;
  kr += 6.0 * tmp * vr;
  kr -= 3.0 * tmp * tmp;
  vr -= tmp;
  return kr / (vr * vr);
}

extern inline double normalized_kurtosis(double *x, int n) {
  return kurtosis(x, n) / 3.0;
}

#ifdef DEBUG
int main() {
  int N = 7;
  double x[] = {1.0, 2.0, 3.0, 1.1, 1.2, 1.3, 2.4};
  double res = kurtosis(x, N);
  printf("Fourth moment of x: %f\n", fourth_moment(x, N));
  printf("Fourth central moment of x: %f\n", fourth_central_moment(x, N));
  printf("Kurtosis of x: %f\n", res);
  res = normalized_kurtosis(x, N);
  printf("Normalized kurtosis of x: %f\n", res);
  return 0;
}
#endif
