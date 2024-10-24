#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double * boot_w_repl(double *y, unsigned n, unsigned max_iter, double (*estimator)(double *, unsigned)) {
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
	if (x) {
		res = 0.0;
		for (; i < n; i++) res += x[i];
		res /= (double) n;
	}
	return res;
}

#define N 10
#define B 1000

int main() {
	double yvec[] = {10.0, 2.01, 1.2, -5.1, -6.7, -9.0, 12.12, 5.2, -4.1, -6.7 };
	double *boot_means;
	srand(time(NULL));
	boot_means = boot_w_repl(yvec, N, B, mean_4_boot);
	if (boot_means) {
		printf("Bootstrap (with replacement): Estimated mean is %f\n", mean_4_boot(boot_means, B));
	}
	free(boot_means);
	return 0;
}
#endif
