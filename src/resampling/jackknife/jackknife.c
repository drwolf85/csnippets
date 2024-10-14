#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double jackknife_mean(double *x, size_t n) {
	size_t i;
	double mx = 0.0;
	double *xbars = (double *) malloc(n * sizeof(double));
	double const inm1 = 1.0 / (double) (n - 1);
	if (x && xbars && n > 1) {
		for (i = 0; i < n; i++) mx += x[i];
		for (i = 0; i < n; i++) xbars[i] = inm1 * (mx - x[i]);
		mx = *xbars;
		for (i = 1; i < n; i++) mx += xbars[i];
		mx /= (double) n;
	}
	free(xbars);
	return mx;
}

double jackknife_mean_var(double *x, size_t n) {
	size_t i;
	double mx = 0.0;
	double vr = 0.0;
	double *xbars = (double *) malloc(n * sizeof(double));
	double const inm1 = 1.0 / (double) (n - 1);
	if (x && xbars && n > 1) {
		for (i = 0; i < n; i++) mx += x[i];
		for (i = 0; i < n; i++) xbars[i] = inm1 * (mx - x[i]);
		mx = *xbars;
		vr = mx * mx;
		for (i = 1; i < n; i++) {
			mx += xbars[i];
			vr += xbars[i] * xbars[i];
		}
		mx /= (double) n;
		vr /= (double) n;
	}
	free(xbars);
	return (vr - mx * mx) / (double) (n - 1);
}

double * pseudo_values(double *x, size_t n, double (*estimator)(double *, size_t)) {
	size_t const nm1 = n - 1;
	size_t i, j;
	double *res = (double *) calloc(n, sizeof(double));
	double *tmpx = (double *) calloc(nm1, sizeof(double));
	double E;
	if (tmpx && res && x && estimator) {
		E = estimator(x, n);
		for (j = 0; j < n; j++) {
			for (i = 0; i < nm1; i++) tmpx[i] = x[i + (i >= j)];
			res[j] = (double) n * E - (double) nm1 * estimator(tmpx, nm1);
		}
	}
	free(tmpx);
	return res;
}

double jackknife_estimate(double *x, size_t n, double (*estimator)(double *, size_t)) {
	size_t i;
	double mx = nan("");
	double *psv = pseudo_values(x, n, estimator);
	if (psv && x) {
		mx = *psv;
		for (i = 1; i < n; i++) {
			mx += psv[i];
		}
		mx /= (double) n;
	}
	free(psv);
	return mx;
}

double jackknife_estimate_var(double *x, size_t n, double (*estimator)(double *, size_t)) {
	size_t i;
	double vr = nan("");
	double mx = nan("");
	double *psv = pseudo_values(x, n, estimator);
	if (psv && x) {
		mx = *psv;
		vr = *psv * *psv;
		for (i = 1; i < n; i++) {
			mx += psv[i];
			vr += psv[i] * psv[i];
		}
		mx /= (double) n;
		vr /= (double) n;
		vr -= mx * mx;
	}
	free(psv);
	return vr / (double) (n - 1);
}

#ifdef DEBUG
double myvar(double *x, size_t n) {
	double mx = nan("");
	double vr = nan("");
	size_t i;
	if (x) {
		vr = *x * *x;
		mx = *x;
		for (i = 1; i < n; i++) {
			vr += x[i] * x[i];
			mx += x[i];
		}
		vr -= mx * mx / (double) n;
	}
	return vr / (double) n; /* Biased variance estimator */
}

int main() {
	double m = 0.0;
	double x[7] = {1.02, 0.98, 0.85, 1.11, 1.05, 0.92, 0.88};
	int i;
	for (i = 0; i < 7; i++) m += x[i];
	printf("Mean of x: %f; and biased Variance of x: %f\n", m / 7.0, myvar(x, 7));
	printf("Jackknife mean: %f\n", jackknife_mean(x, 7));
	printf("Variance of jackknife mean: %f\n", jackknife_mean_var(x, 7));
	printf("Jackknife Var(x): %f\n", jackknife_estimate(x, 7, myvar));
	printf("Variance of jackknife Var(x): %f\n", jackknife_estimate_var(x, 7, myvar));
	return 0;
}
#endif

