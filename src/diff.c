#include <math.h>
#include <stdlib.h>
#ifdef DEBUG
#include <stdio.h>
#endif

static double choose(int n, int k) {
	double res = lgamma((double) (n + 1));
	res -= lgamma((double) (k + 1));
	res -= lgamma((double) (n - k + 1)); 
	return exp(res);
}

/**
 @brief Difference of order `d`
 @param x pointer to a vector of `double`s
 @param n number of components in `x`
 @param d order of the difference 
 		  (as in the $\delta^d$ operator in time-series)
 @return a pointer to the vector containing the 
 		 differences of order `d` of the vector `x`
 */
double * diff(double *x, int n, int d) {
	int i, j;
	double *r, *filter;
	
	if (d >= n || d < 0) return NULL;
	if (d == 0) {
		r = (double *) malloc(n * sizeof(double));
		if (r) for (i = 0; i < n; i++) r[i] = x[i];
		return r;
	}
	r = (double *) calloc(n - d, sizeof(double));
	filter = (double *) malloc((d + 1) * sizeof(double));
	if (r && filter) {
		j = 0;
		while (j <= d) {
			filter[j] = choose(d, j);
			j++;
			filter[j - 1] *= 2.0 * (double) (j & 1) - 1.0;
		}
		for (i = 0; i < n - d; i++) {
			for (j = 0; j <= d; j++)
				r[i] += x[i+j] * filter[j];
		}
	}
	free(filter);
	return r;
}

#ifdef DEBUG
int main() {
	int const N = 8;
	int const D = 2;
	int i;
	double test_x[] = {0.98, 0.05, -0.51, -0.25, 0.12, -0.75, 0.89, 0.11};
	double *df;
	printf("pointer before processing %p\n", df);
	df = diff(test_x, N, D);
	printf("pointer after processing %p\n", df);
	printf("x\n");
	for (i = 0; i < N; i++) printf("%.2f ", test_x[i]);
	printf("\ndiff\n");
	for (i = 0; i < N - D; i++) printf("%.2f ", df[i]);
	printf("\n");
	free(df);
}
#endif

