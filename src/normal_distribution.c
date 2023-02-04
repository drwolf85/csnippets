#include <stdio.h>
#include <math.h>

/**
 * The function `dnorm` returns the value of the normal density function at `x` with mean `m` and
 * standard deviation `s`
 * 
 * @param x the value we're evaluating the PDF at
 * @param m mean
 * @param s standard deviation
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dnorm(double x, double m, double s) {
    double z = x - m;
    z /= s; 
    z = exp(-0.5 * z * z);
    z /= s * sqrt(2.0 * M_PI);
    return z;
}

/**
 * The function `pnorm` returns the probability that a random variable from a normal distribution
 * with mean `m` and standard deviation `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m mean
 * @param s standard deviation
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double pnorm(double x, double m, double s) {
    double z = x - m;
    z /= s * sqrt(2.0); 
    z = 0.5 + 0.5 * erf(z);
    return z;
}

/**
 * It takes a probability, a mean, and a standard deviation, and returns the value of the normal
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m mean
 * @param s standard deviation
 * 
 * @return The quantile function of the normal distribution.
 */
double qnorm(double p, double m, double s) {
    double old;
    double z = 0.25 * log(p / (1.0 - p)); /* Initial approximation */
    double sdv = dnorm(z, 0.0, 1.0);
    do {
        old = z;
        z += (p - pnorm(z, 0.0, 1.0)) / sdv;
        sdv = dnorm(z, 0.0, 1.0);
    } while (sdv > 1e-9 && fabs(old - z) > 1e-12);
    return s * z + m;
}

int main() {
    double x = -1.64;
    double d, p, q;
    d = dnorm(x, 0.0, 1.0);
    p = pnorm(x, 0.0, 1.0);
    q = qnorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    return 0;
}

