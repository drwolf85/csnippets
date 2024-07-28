#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double univar_mgf(double t, double *x, int n) {
	double res = 0.0;
	int i;
	for (i = 0; i < n; i++) {
		res += exp(x[i] * t);
	}
	return res / (double) n;
}

double multivar_mgf(double *t, double *x, int n, int p) {
	double res = 0.0, tmp;
	int i, j;
	/* Assuming matrix `x` is stored in memory in row-major format */
	#pragma omp parallel for simd private(i, j, tmp) reduction(+ : res)
	for (i = 0; i < n; i++) {
		tmp = 0.0;
		for (j = 0; j < p; j++) {
			tmp += x[p * i + j] * t[j];
		}
		res += exp(tmp);
	}
	return res / (double) n;
}

#ifdef DEBUG
int main() {
	double x[] = {1.2, 0.3, -1.4};
	double t[] = {1.0, 10, 1.0};
	double mgf = univar_mgf(1.0, x, 3);
	double mgf3 = multivar_mgf(t, x, 1, 3);
	printf("Univar. MGF = %f\n"
	       "Multivar. MGF = %f\n", mgf, mgf3);
	return 0;
}
#endif

