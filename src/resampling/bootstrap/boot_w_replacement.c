#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * @brief Bootstrap with replacement
 * 
 * @param y Pointer to a vector of data
 * @param n Length of the vector `y`
 * @param max_iter Maximum number of bootstrap iterations
 * @param estimator Pointer to a function that produces estimates for a parameter of interest
 * @return double* 
 */
extern double * boot_w_repl(double *y, unsigned n, unsigned max_iter, double (*estimator)(double *, unsigned)) {
	unsigned b, i;
	double *repl, *smp;
	repl = (double *) calloc(max_iter, sizeof(double));
	smp = (double *) malloc(n * sizeof(double));
	if (y && estimator && smp && repl) {
		for (b = 0; b < max_iter; b++) {
			for (i = 0; i < n; i++)
				smp[i] = y[(unsigned) (rand() % n)];
			repl[b] = estimator(smp, n);
		}
	}
	free(smp);
	return repl;
}

#ifdef DEBUG
double mean_4_boot(double *x, unsigned n) {
	double res = nan("");
	unsigned i = 0;
	if (x && n > 0) {
		res = 0.0;
		for (; i < n; i++) res += x[i];
		res /= (double) n;
	}
	return res;
}

double var_4_boot(double *x, unsigned n) {
	double res = nan("");
	double mx;
	unsigned i = 0;
	if (x && n > 1) {
		res = 0.0;
		mx = 0.0;
		for (; i < n; i++) {
			res += x[i] * x[i];
			mx += x[i];
		}
		res -= mx * mx / (double) n;
		res /= (double) (n - 1);
	}
	return res;
}

#define N 961
#define B 10000

int main() {
	unsigned i;
	double *yvec;
	double *boot_means;
	srand(time(NULL));
	yvec = (double *) calloc(N, sizeof(double));
	if (yvec) {
		for (i = 0; i < N; i++) yvec[i] = (0.5 + (double) rand()) / (1.0 + (double) RAND_MAX);
		boot_means = boot_w_repl(yvec, N, B, mean_4_boot);
		printf("Bootstrap (with replacement):\n");
		if (boot_means) printf("\tEstimated mean is %f\n", mean_4_boot(boot_means, B));
		free(boot_means);
		boot_means = boot_w_repl(yvec, N, B, var_4_boot);
		if (boot_means) printf("\tEstimated variance is %f\n", mean_4_boot(boot_means, B));
		free(boot_means);
	}
	free(yvec);
	return 0;
}
#endif
