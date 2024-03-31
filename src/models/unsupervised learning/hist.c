#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define MY_EPS_RNG 1e-9

typedef struct lik_obj {
  double v;
  uint32_t i;
} LK;

static inline double xlogx(uint32_t x, uint32_t b, uint32_t n) {
  double res = 0.0;
  if (x) res = (double) x * log((double) (x * b) / (double) n);
  return res;
}

double pen_lik(uint32_t b, double *x_sorted, uint32_t n) {
  /* Based on Birge and Rozenholc (2006) */
  double mn, mx, w, tmp, res = nan("");
  uint32_t i, j, tof, ni;
  if (n > 1 && b > 1) {
    res = 0.0;
    i = j = ni = 0;
    mn = x_sorted[0] - MY_EPS_RNG;
    mx = x_sorted[n - 1] + MY_EPS_RNG;
    w = mx - mn;
    w = 1.0 / w;
    /* Likelihood */
    for(; i < n; i++) {
      tmp = x_sorted[i] - mn;
      tmp *= (double) b * w;
      tof = (j == (uint32_t) (tmp));
      ni += tof;
      if(!tof) {
        res += xlogx(ni, b, n);
        ni = 1;
        j++;
      }
    }
    res += xlogx(ni, b, n);
    /* Penalty */
    tmp = log((double) b);
    w = sqrt(tmp);
    tmp *= tmp;
    b--;
    res -= b + tmp * w;
  }
  return res;
}

int cmp_dbl(void const *aa, void const *bb) {
  double a = *(double *) aa;
  double b = *(double *) bb;
  return (int) (a >= b) * 2 - 1;
}

int cmp_plk(void const *aa, void const *bb) {
  double a = *(double *) aa;
  double b = *(double *) bb;
  return (int) (a < b) * 2 - 1;
}

double * get_hist(double *x_sorted, uint32_t n, uint32_t b) {
  uint32_t i;
  double mn, mx, w, d;
  double *H = NULL;
  H = (double *) calloc(b, sizeof(double));
  if (H) {
    mn = x_sorted[0] - MY_EPS_RNG;
    mx = x_sorted[n - 1] + MY_EPS_RNG;
    w = mx - mn;
    w = (double) b / w;
    d = 1.0 / (w * (double) n);
    for (i = 0; i < n; i++) {
      H[(uint32_t)(w * (x_sorted[i] - mn))] += d;
    }
  }
  return H;
}

#ifdef DEBUG
#define N 100

int main() {
/* N 30  double x[] = {-4.935813, -5.327745, -2.341657, -5.76643, -2.19953, -2.584017, -3.667758, -2.644171, -3.4779, -4.421868, 1.582398, 0.1485875, -0.5055712, 1.239271, 0.2881622, 1.307816, 1.100757, 0.1301804, -1.062798, 0.2458724, 7.841751, 7.244247, 5.476391, 4.343381, 5.669445, 5.737663, 4.916479, 4.898192, 6.777139, 5.701327}; */
  double x[] = {-4.479334, -3.486911, -4.663997, -3.889537, -2.948605, -4.224565, -3.387195, -2.784865, -4.893749, -2.808751, -4.919912, -4.795104, -5.917237, -4.874074, -5.422464, -4.804668, -2.653443, -3.87883, -3.705247, -2.833873, -3.231187, -3.592668, -4.811681, -3.543349, -3.206718, -5.307814, -3.85983, -3.825077, -4.234471, -5.097686, -1.231694, -1.211571, 1.734895, 1.923577, -1.503239, 1.792012, -0.2362421, -0.1270063, 0.3496261, 0.476487, 0.002978298, 0.4081354, -0.3505237, -1.266738, 1.982123, -0.6330949, -0.9996398, 0.7744802, 1.032079, -0.9791578, 0.9042083, -0.4911959, -1.138773, -0.6428232, -0.08041188, -0.7114716, 0.7805432, 0.8442178, 0.05498607, 1.366914, 4.152605, 5.598348, 4.117127, 3.967081, 3.564966, 5.265716, 5.083035, 5.172237, 4.077041, 2.438205, 2.608562, 3.492392, 3.80316, 5.391267, 3.744586, 3.540183, 3.307869, 5.233534, 4.671848, 3.343961, 1.72468, 3.987106, 1.781157, 3.388748, 4.657545, 3.230201, 6.973001, 2.801573, 4.559738, 4.623309, 4.338109, 4.588389, 3.421892, 3.417098, 4.117501, 2.153411, 3.7875, 4.066211, 4.084818, 3.912522};
  double xsrt[N] = {0};
  double *hist;
  LK plk[N] = {0};
  uint32_t i;
  
  for (i = 0; i < N; i++) xsrt[i] = x[i];
  qsort(xsrt, N, sizeof(double), cmp_dbl);
  printf("Sorted data:\n");
  for (i = 0; i < N; i++) printf("%.2f ", xsrt[i]);
  printf("\nLikelihoods by bin numbers:\n");
  for (i = 2; i < N; i++) {
    plk[i].i = i;
	  printf("lik %f for b = %u\n", plk[i].v = pen_lik(i, xsrt, N), i);
	}
  qsort(plk+2, N-2, sizeof(LK), cmp_plk);
  printf("Sorted liks:\n");
  for (i = 2; i < N; i++) printf("%.3f ", plk[i].v);
  printf("\n");
  printf("Best number of bins: %u\n", plk[2].i);
  hist = get_hist(xsrt, N, plk[2].i);
  if (hist) {
    printf("Histogram densities:\n");
    for (i = 0; i < plk[2].i; i++) printf("%f ", hist[i]);
    printf("\n");
  }
  free(hist);
  return 0;
}
#endif

