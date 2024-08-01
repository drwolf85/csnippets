#include <stdio.h>
#include <math.h>
#include <omp.h>

double wt_third_moment(double *x, double *w, int n) {
  int i;
  double sk = 0.0, nnan = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : sk, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i]) && !isnan(w[i])) {
      tmp = x[i] * x[i] * x[i];
      nnan += w[i];
      sk += tmp * w[i];
    }
  }
  sk /= nnan;
  return sk;
}

double wt_third_central_moment(double *x, double *w, int n) {
  int i;
  double mu = 0.0, vr = 0.0, sk = 0.0, nnan = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i]) && !isnan(w[i])) {
      tmp = x[i];
      mu += tmp * w[i];
      tmp *= x[i];
      vr += tmp * w[i];
      tmp *= x[i];
      sk += tmp * w[i];
      nnan += w[i];
    }
  }
  mu /= nnan;
  vr /= nnan;
  sk /= nnan;
  sk -= 3.0 * vr * mu;
  sk += 2.0 * mu * mu * mu;
  return sk;
}

double wt_skewness(double *x, double *w, int n) {
  int i;
  double mu = 0.0, sk = 0.0, vr = 0.0, nnan = 0.0, tmp;
  #pragma omp parallel for simd private(i, tmp) reduction(+ : mu, vr, sk, nnan)
  for (i = 0; i < n; i++) {
    if (!isnan(x[i]) && !isnan(w[i])) {
      tmp = x[i];
      mu += tmp * w[i];
      tmp *= x[i];
      vr += tmp * w[i];
      tmp *= x[i];
      sk += tmp * w[i];
      nnan += w[i];
    }
  }
  mu /= nnan;
  vr /= nnan;
  sk /= nnan;
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
  double w[] = {1.0, 2.0, 1.0, 0.5, 3.0, 3.14, 4.0};
  double res = wt_skewness(x, w, N);
  printf("Weighted third moment of x: %f\n", wt_third_moment(x, w, N));
  printf("Weighted third central moment of x: %f\n", wt_third_central_moment(x, w, N));
  printf("Weighted skewness of x: %f\n", res);
  return 0;
}
#endif
