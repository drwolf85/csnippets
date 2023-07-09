#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function `dlaplace` returns the value of the Laplace density function at `x` 
 * with location parameter `m` and scale parameter `s` 
 * 
 * @param x the value we're evaluating the PDF at
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dlaplace(double x, double m, double s) {
    double z = m - x;
    s = 1.0 / s; 
    z = exp(-2.0 * fabs(z * s)) * s;
    return z;
}

/**
 * The function `plaplace` returns the probability that a random variable from a Laplace distribution
 * with location parameter `m` and scale parameter `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double plaplace(double x, double m, double s) {
    double z = m - x;
    z /= s; 
    z = exp(-2.0 * fabs(z));
    z *= 0.5 * (x < m) - 1.0 * (x >= m);
    z += (double) (x >= m);
    return z;
}

/**
 * It takes a probability, a location, and a scale parameter, and returns the value of the Laplace
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The quantile function of the Laplace distribution.
 */
double qlaplace(double p, double m, double s) {
    double z = nan("");
    double sgn = 2.0 * (double) (p < 0.5) - 1.0;
    if (p >= 0.0 && p <= 1.0 && s >= 0.0) 
        z = 2.0 * fabs(p - 0.5);
    return s * sgn * log(1.0 - z) + m;
}

/** 
 * The function rlaplace() is a C function that generates a random number from a Laplace distribution with
 * location parameter `mu` and scale parameter `sd`
 * 
 * @param mu location parameter of the Laplace distribution
 * @param sd scale parameter of the Laplace distribution
 * 
 * @return A random number from a Laplace distribution with location `mu` and scale `sd`.
 */
double rlaplace(double mu, double sd) {
   unsigned long u, m = ~(1 << 31);
   double a, b, s;
   u = rand();
   u &= m;
   return qlaplace(ldexp((double) u, -31), mu, sd);
}

/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dlaplace(x, 0.0, 1.0);
    p = plaplace(x, 0.0, 1.0);
    q = qlaplace(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a Laplace variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rlaplace(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
