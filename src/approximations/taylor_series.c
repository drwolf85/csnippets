#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define H_SEP 0.01

static double deriv(double x0, double h, double htype, double(*f)(double), unsigned p, double scaling) {
    double res = 0.0;
    double xl, xr;
    if (p > 0) {
	    xl = xr = x0;
	    xr += h * (1.0 - htype);
	    xl -= h * htype;
	    res = deriv(xr, scaling * h, htype, f, p - 1, scaling);
	    res -= deriv(xl, scaling * h, htype, f, p - 1, scaling);
        return res / h;
    }
    else {
        return f(x0);
    }
}

extern double * mac_laurin_expand(unsigned p, double (*f)(double)) {
	unsigned i;
	double *coef = (double *) calloc((p + 1), sizeof(double));
	double den = 1.0;
	if (coef) {
		coef[0] = f(0.0) / den;
		for (i = 1; i <= p; i++) {
			coef[i] = deriv(0.0, 1e-2, 0.5, f, i, 1.0);
			den *= (double) i;
			coef[i] /= den;
		}
	}
	return coef;
}

extern double eval_mac_laurin(double x, unsigned p, double *coef) {
    unsigned i;
    double res = 0.0, tmp = 1.0;
    for (i = 0; i < p; i++) {
        res += tmp * coef[i];
        tmp *= x;
    }
    return res + tmp * coef[i];
}

extern double * taylor_expand(double x0, unsigned p, double (*f)(double)) {
	unsigned i;
	double *coef = (double *) calloc((p + 1), sizeof(double));
	double den = 1.0;
	if (coef) {
		coef[0] = f(x0) / den;
		for (i = 1; i <= p; i++) {
			coef[i] = deriv(x0, 1e-2, 0.5, f, i, 1.0);
			den *= (double) i;
			coef[i] /= den;
		}
	}
	return coef;
}

extern double eval_taylor(double x, unsigned p, double *coef, double x0) {
    unsigned i;
    double const df = x - x0;
    double res = 0.0, tmp = 1.0;
    for (i = 0; i < p; i++) {
        res += tmp * coef[i];
        tmp *= df;
    }
    return res + tmp * coef[i];
}

#ifdef DEBUG
int main() {
    unsigned i;
    unsigned const power = 4;
    double const x_0 = sqrt(M_PI) * 0.5;
    double *cvec = mac_laurin_expand(power, exp);
    if (cvec) {
        printf("The polynom based on Mac Laurin approximation is:\n");
        printf("%.3f ", cvec[0]);
        for (i = 1; i <= power; i++) {
            printf("+ %.3f * x^%d ", cvec[i], i);
        }
        printf("\n");
        printf("Evaluation at x = 1: %f\n\n", eval_mac_laurin(1.0, power, cvec));
    }    
    free(cvec);
    cvec = taylor_expand(x_0, power, exp);
    if (cvec) {
        printf("The polynom based on Taylor approximation is:\n");
        printf("%.3f ", cvec[0]);
        for (i = 1; i <= power; i++) {
            printf("+ %.3f * (x - %.3f)^%d ", cvec[i], x_0, i);
        }
        printf("\n");
        printf("Evaluation at x = 1: %f\n\n", eval_taylor(1.0, power, cvec, x_0));
    }
    free(cvec);
    printf("True value at x = 1: %f\n", exp(1.0));
    return 0;
}
#endif

