#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double cubic_basis(double x, double knot) {
	double r = x - knot;
	r *= (double) (r > 0.0);
	r *= r * r;
	return r;
}

#ifdef DEBUG
#define N 10
#define K 5
#define STEP (1.0 / (double) (N))

int main(void) {
	int i, j;
	double knt[K];
	double par[K + 4];
	double x[N];
	double y[N] = {0};
	double tmp;
	srand(time(NULL));
	x[0] = 0.5 * STEP;
	for (i = 1; i < N; i++) x[i] = x[i - 1] + STEP;
	for (i = 0; i < K; i++) knt[i] = (0.5 + (double) rand()) / (double) RAND_MAX;
	i = 0;
	par[i] = 2.0 * (0.5 + (double) rand()) / (double) RAND_MAX - 1.0;
	for (i = 1; i < K + 4; i++) {
		par[i] = 2.0 * (0.5 + (double) rand()) / (double) RAND_MAX - 1.0;
	}
	for (i = 0; i < N; i++) {
		tmp = x[i];
		y[i] = par[0] + par[1] * tmp;
		tmp *= tmp;
		y[i] += par[2] * tmp;
		tmp *= tmp;
		y[i] += par[3] * tmp;
		for (j = 4; j < K + 4; j++) {
			y[i] += par[j] * cubic_basis(x[i], knt[j - 4]);
		}
	}
	printf("Testing Inputs and Outputs\n");
	printf("x <- c("); for (i = 0; i < N - 1; i++) printf("%g, ", x[i]);
	printf("%g)\n", x[i]);
	printf("y <- c("); for (i = 0; i < N - 1; i++) printf("%g, ", y[i]);
	printf("%g)\nplot(x, y)\n\n", y[i]);
	printf("Testing Knots\n");
	for (i = 0; i < K; i++) printf("%g ", knt[i]); printf("\n");
	return 0;
}
#endif

