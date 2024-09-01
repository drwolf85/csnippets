#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Numerical approximation of the $p$-th derivative of $f(x)$ computed in $x = x_0$
 * 
 * @param x0 Value where to compute the second derivative of $f$
 * @param h Positive number for the length of the segment in $\mathbb{R}_+$
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}\to\mathbb{R}$
 * @param p Degree of the derivative (assumed to be a non-negative integer)
 * @return double 
 */
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

#ifdef DEBUG
int main() {
    printf("Fourth derivative of exp(x) in x = 1: %f\n", deriv(1.0, 1e-2, 0.5, exp, 4));
    return 0;
}
#endif

