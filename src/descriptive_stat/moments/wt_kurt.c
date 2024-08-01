#include <stdio.h>
#include <math.h>
#include <omp.h>

extern inline double wt_fourth_moment(double *x, double *w, int n) {
  int i;
  double kr = 0.0, nnan = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : kr, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i]) && !isnan(w[i])) {
      tmp = x[i];
      tmp *= x[i];
      tmp *= x[i];
      tmp *= x[i];
      kr += tmp * w[i];
      nnan += w[i];
    }
  }
  kr /= nnan;
  return kr;
}

extern inline double wt_fourth_central_moment(double *x, double *w, int n) {
  int i;
  double mu = 0.0, vr = 0.0, sk = 0.0, kr = 0.0, nnan = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, kr, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i]) && !isnan(w[i])) {
      tmp = x[i];
      mu += tmp * w[i];
      tmp *= x[i];
      vr += tmp * w[i];
      tmp *= x[i];
      sk += tmp * w[i];
      tmp *= x[i];
      kr += tmp * w[i];
      nnan += w[i];
    }
  }
  mu /= nnan;
  vr /= nnan;
  sk /= nnan;
  kr /= nnan;
  kr -= 4.0 * mu * sk;
  tmp = mu * mu;
  kr += 6.0 * tmp * vr;
  kr -= 3.0 * tmp * tmp;
  return kr;
}

extern inline double wt_kurtosis(double *x, double *w, int n) {
  int i;
  double mu = 0.0, sk = 0.0, vr = 0.0, kr = 0.0, nnan, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, kr, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(w[i]) && !isnan(x[i])) {
      tmp = x[i];
      mu += tmp * w[i];
      tmp *= x[i];
      vr += tmp * w[i];
      tmp *= x[i];
      sk += tmp * w[i];
      tmp *= x[i];
      kr += tmp * w[i];
      nnan += w[i];
    }
  }
  mu /= nnan;
  vr /= nnan;
  sk /= nnan;
  kr /= nnan;
  kr -= 4.0 * mu * sk;
  tmp = mu * mu;
  kr += 6.0 * tmp * vr;
  kr -= 3.0 * tmp * tmp;
  vr -= tmp;
  return kr / (vr * vr);
}

extern inline double wt_normalized_kurtosis(double *x, double *w, int n) {
  return wt_kurtosis(x, w, n) / 3.0;
}

#ifdef DEBUG
int main() {
  int N = 7;
  double x[] = {1.0, 2.0, 3.0, 1.1, 1.2, 1.3, 2.4};
  double w[] = {1.0, 2.0, 1.0, 0.5, 3.0, 3.14, 4.0};
  double res = wt_kurtosis(x, w, N);
  printf("Weighted fourth moment of x: %f\n", wt_fourth_moment(x, w, N));
  printf("Weighted fourth central moment of x: %f\n", wt_fourth_central_moment(x, w, N));
  printf("Weighted kurtosis of x: %f\n", res);
  res = wt_normalized_kurtosis(x, w, N);
  printf("Weighted normalized kurtosis of x: %f\n", res);
  return 0;
}
#endif
