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
 * @param scaling Positive number in $\mathbb{R}_+$ representing a scaling factor for the segment `h`
 * @return double 
 */
double deriv(double x0, double h, double htype, double(*f)(double), unsigned p, double scaling) {
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

#ifdef DEBUG
int main() {
    printf("Fourth derivative of exp(x) in x = 1: %f (Exact)\n", exp(1.0));
    printf("Fourth derivative of exp(x) in x = 1: %f (Approx. w/ scaling=1)\n", deriv(1.0, 1e-2, 0.5, exp, 4, 1.0));
    printf("Fourth derivative of exp(x) in x = 1: %f (Approx. w/ scaling=0.475)\n", deriv(1.0, 1e-2, 0.5, exp, 4, 0.475));
    return 0;
}
#endif

