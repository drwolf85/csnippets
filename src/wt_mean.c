#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

double weighted_mean(double *x, double *w, uint32_t n) {
  uint32_t i;
  double sm, res = nan("");
  if (n > 0) {
    res = sm = 0.0;
    for (i = 0; i < n; i++) {
      sm += w[i];
      res += w[i] * x[i];
    }
    res /= sm;
  }
  return res;
}

