#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

complex univar_charfun(double t, double *x, int n) {
	complex res = 0.0 + I * 0.0;
	int i;
	for (i = 0; i < n; i++) {
		res += cexp(x[i] * t * I);
	}
	return res / (double) n;
}

complex multivar_charfun(double *t, double *x, int n, int p) {
	complex res = 0.0 + I * 0.0;
	double tmp;
	int i, j;
	/* Assuming matrix `x` is stored in memory in row-major format */
	#pragma omp parallel for simd private(i, j, tmp) reduction(+ : res)
	for (i = 0; i < n; i++) {
		tmp = 0.0;
		for (j = 0; j < p; j++) {
			tmp += x[p * i + j] * t[j];
		}
		res += cexp(tmp * I);
	}
	return res / (double) n;
}

#ifdef DEBUG
int main() {
	double x[] = {1.2, 0.1, -1.4};
	double t[] = {1.0, 1.0, 1.0};
	complex mgf = univar_charfun(1.0, x, 3);
	complex mgf3 = multivar_charfun(t, x, 1, 3);
	printf("Univar. Char.Fun. = %f%s%fi\n"
	       "Multivar. Char.Fun. = %f%s%fi\n", \
	       creal(mgf), cimag(mgf) < 0.0 ? "" : "+", \
	       creal(mgf3), cimag(mgf3) < 0.0 ? "" : "+");
	return 0;
}
#endif
