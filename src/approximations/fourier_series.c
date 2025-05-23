#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_STEPS 10000

extern double * fourier_expand(unsigned p, double (*f)(double)) {
	unsigned i;
	double tmp, x = 0.0;
	double const inv = 1.0 / (double) N_STEPS;
	double const step = 2.0 * M_PI * inv;
	double *coef = (double *) calloc((1 + (p << 1)), sizeof(double));
	if (coef) {
		while (x < M_PI * 2.0) {
			coef[0] += f(x) * inv;
			x += step;
		}
		for (i = 0; i < p; i++) {
			x = 0.0;
			while (x < M_PI * 2.0) {
				tmp = x * (double) (i + 1);
				coef[1 + i] += inv * f(x) * sin(tmp);
				coef[1 + p + i] += inv * f(x) * cos(tmp);
				x += step;
			}
		}
	}
	return coef;
}

extern double eval_fourier(double x, unsigned p, double *coef) {
    unsigned i;
    double tmp, res = *coef;
    for (i = 0; i < p; i++) {
        tmp = x * (double) (i + 1);
	res += sin(tmp) * coef[i + 1];
	res += cos(tmp) * coef[i + 1 + p];
    }
    return res;
}

#ifdef DEBUG
static double mylogitan(double x) {
	return 1.0 / (1.0 + exp(-tan(x)));
}

int main() {
    unsigned i;
    unsigned const order = 8;
    double *cvec = fourier_expand(order, mylogitan);
    if (cvec) {
        printf("The Fourier approximation is:\n");
        printf("%.3g ", cvec[0]);
        for (i = 1; i <= 2 * order; i++) {
            printf("+ %.3g * %s(x * %d) ", cvec[i], i & 1 ? "sin" : "cos", (i + 1) >> 1);
        }
        printf("\n");
        printf("Evaluation at x = 0.18: %f\n\n", eval_fourier(0.18, order, cvec));
    }    
    if (cvec) free(cvec);
    printf("True value at x = 0.18: %f\n", mylogitan(0.18));
    return 0;
}
#endif

