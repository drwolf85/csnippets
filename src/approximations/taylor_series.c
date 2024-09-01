#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define H_SEP 0.01

double deriv(double x0, double h, double htype, double(*f)(double), unsigned p) {
    double res = 0.0;
    double xl, xr;
    if (p > 0) {
	    xl = xr = x0;
	    xr += h * (1.0 - htype);
	    xl -= h * htype;
	    res = deriv(xr, h, htype, f, p - 1);
        res -= deriv(xl, h, htype, f, p - 1);
        return res / h;
    }
    else {
        return f(x0);
    }
}

double * taylor_expand(double x0, unsigned p, double (*f)(double)) {
	unsigned i;
	double *coef = (double *) calloc((p + 1), sizeof(double));
	double den = 1.0;
	if (coef) {
		coef[0] = f(x0) / den;
		for (i = 1; i <= p; i++) {
			coef[i] = deriv(x0, 1e-2, 0.5, f, i);
			den *= (double) i;
			coef[i] /= den;
		}
	}
	return coef;
}

#ifdef DEBUG
int main() {
    unsigned i;
    unsigned const power = 4;
    double *cvec = taylor_expand(sqrt(M_PI) * 0.5, power, exp);
    if (cvec) {
        printf("The polynom based on Taylor approximation is:\n");
        printf("%f ", cvec[0]);
        for (i = 1; i <= power; i++) {
            printf("+ %f x^%d ", cvec[i], i);
        }
        printf("\n");
    }
    free(cvec);
    return 0;
}
#endif

