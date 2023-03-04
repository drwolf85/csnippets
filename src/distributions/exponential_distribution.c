#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * If the rate parameter is non-negative, then the function returns the density of the exponential
 * distribution with that rate parameter, otherwise it returns NaN
 * 
 * @param x the value at which to evaluate the density
 * @param lambda the rate parameter of the distribution.
 * 
 * @return The probability density function of the exponential distribution.
 */
double dexp(double x, double lambda) {
    double z = nan("");
    if (lambda >= 0.0) {
        z = (double) (x >= 0.0) * lambda * exp(-lambda * x);
    }
    return z;
}

/**
 * `pexp` returns the probability that an exponential random variable with rate `lambda` is less than
 * or equal to `x`
 * 
 * @param x the value at which to evaluate the distribution
 * @param lambda the rate parameter of the exponential distribution
 * 
 * @return The probability of a random variable being less than or equal to `x`.
 */
double pexp(double x, double lambda) {
    double z = nan("");
    if (lambda >= 0.0) {
        z = (double) (x > 0.0) * (1.0 - exp(-lambda * x));
    }
    return z;
}

/**
 * It returns the quantile of the exponential distribution with rate `lambda` at probability `p`.
 * 
 * @param p the probability of the event occurring
 * @param lambda the rate parameter of the distribution.
 * 
 * @return The quantile function of the exponential distribution.
 */
double qexp(double p, double lambda) {
    double z = nan("");
    if (lambda >= 0.0 && p >= 0.0 && p <= 1.0) {
        z = -log1p(-p) / lambda;
    }
    return z;
}

/**
 * Generate a random number from an exponential distribution with parameter `lambda` using the
 * inverse transform method
 * 
 * @param lambda the rate parameter of the distribution.
 * 
 * @return a random number from an exponential distribution with a given `lambda`.
 */
double rexp(double lambda) {
    unsigned long u, m;
    if (lambda >= 0.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        return (31 * M_LN2 - log(u)) / lambda;
    } 
    else {
        return nan("");
    }
}

/* Test function */
int main() {
    double x = 1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dexp(x, 2.0);
    p = pexp(x, 2.0);
    q = qexp(0.95, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a normal variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rexp(2.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
