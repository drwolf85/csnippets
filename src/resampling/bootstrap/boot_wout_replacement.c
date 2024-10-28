#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * @brief Random permutation of an `unsigned char` vector 
 * 
 * @param vec Pointer to the `unsigned char` vector to permute
 * @param n Length of the vector `vec`
 */
static inline void rnd_perm(unsigned char *vec, unsigned n) {
    unsigned i, j;
    for (i = 0; i < n; i++) {
        j = rand() % n;
        vec[j] ^= vec[i];
        vec[i] ^= vec[j];
    }
}

/**
 * @brief Bootstrap without replacement
 * 
 * @param y Pointer to a vector of data
 * @param n Length of the vector `y`
 * @param sub_n Length of the subsample to get at each iteration
 * @param max_iter Maximum number of bootstrap iterations
 * @param estimator Pointer to a function that produces estimates for a parameter of interest
 * @return double* 
 */
extern double * boot_wout_repl(double *y, unsigned const n, unsigned const sub_n, unsigned const max_iter, double (*estimator)(double *, unsigned)) {
	unsigned b, i, j;
	double *repl = NULL, *smp;
    unsigned char *sel;
    if (sub_n < n) {
        repl = (double *) calloc(max_iter, sizeof(double));
        smp = (double *) malloc(sub_n * sizeof(double));
        sel = (unsigned char *) malloc(n * sizeof(char));
        if (y && estimator && sel && smp && repl) {
            for (b = 0; b < max_iter; b++) {
                for (i = 0; i < sub_n; i++) sel[i] = 1;
                for (i = sub_n; i < n; i++) sel[i] = 0;
                rnd_perm(sel, n);
                for (j = 0, i = 0; i < n && j < sub_n; i++, j += (unsigned) sel[i]) 
                    if (sel[i]) smp[j] = y[i];
                repl[b] = estimator(smp, sub_n);
            }
        }
        free(sel);
        free(smp);
    }
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
	unsigned i;
	if (x && n > 1) {
		res = 0.0;
		mx = 0.0;
		for (i = 0; i < n; i++) {
			res += x[i] * x[i];
			mx += x[i];
		}
		res -= mx * mx / (double) n;
		res /= (double) (n - 1);
	}
	return res;
}

#define N 961
#define N_SUB 256
#define B 10000

int main() {
	unsigned i;
	double *yvec;
	double *boot_means;
	srand(time(NULL));
	yvec = (double *) calloc(N, sizeof(double));
	if (yvec) {
		for (i = 0; i < N; i++) yvec[i] = (0.5 + (double) rand()) / (1.0 + (double) RAND_MAX);
		boot_means = boot_wout_repl(yvec, N, N_SUB, B, mean_4_boot);
		printf("Bootstrap (without replacement):\n");
		if (boot_means) printf("\tEstimated mean is %f\n", mean_4_boot(boot_means, B));
		free(boot_means);
		boot_means = boot_wout_repl(yvec, N, N_SUB, B, var_4_boot);
		if (boot_means) printf("\tEstimated variance is %f\n", mean_4_boot(boot_means, B));
	    free(boot_means);
	}
	free(yvec);
	return 0;
}
#endif
