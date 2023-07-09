#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

/** 
 * The function rnorm() is a C function that generates a random number from a normal distribution with
 * mean mu and standard deviation sd
 * 
 * @param mu mean of the normal distribution
 * @param sd standard deviation
 * 
 * @return A random number from a normal distribution with mean mu and standard deviation sd.
 */
double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -30) - 1.0;
   s = a * a;
   b = ldexp((double) v, -30) - 1.0;
   s += b * b * (1.0 - s);
   s = -2.0 * log(s) / s;
   a = b * sqrtf(s);
   return mu + sd * a;
}

/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dnorm(x, 0.0, 1.0);
    p = pnorm(x, 0.0, 1.0);
    q = qnorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a normal variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rnorm(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
