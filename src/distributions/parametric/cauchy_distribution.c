#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function `dcauchy` returns the value of the Cauchy density function at `x` 
 * with location parameter `m` and scale parameter `s` 
 * 
 * @param x the value we're evaluating the PDF at
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dcauchy(double x, double m, double s) {
    double z = nan("");
    x -= m;
    if (s >= 0.0) {
        s = 1.0 / s;
        z = x * s;
    }
    z = 1.0 + z * z;
    return M_1_PI * s / z;
}

/**
 * The function `pcauchy` returns the probability that a random variable from a Cauchy distribution
 * with location parameter `m` and scale parameter `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double pcauchy(double x, double m, double s) {
    double z = nan("");
    x -= m;
    if (s >= 0.0)
        z = x / s;
    return 0.5 + atan(z) * M_1_PI;
}

/**
 * It takes a probability, a location, and a scale parameter, and returns the value of the Cauchy
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The quantile function of the Cauchy distribution.
 */
double qcauchy(double p, double m, double s) {
    double z = nan("");
    if (p >= 0.0 && p <= 1.0 && s >= 0.0) 
        z = M_PI * (p - 0.5);
    z = tan(z);
    return s * z + m;
}

/** 
 * The function rcauchy() is a C function that generates a random number from a Cauchy distribution with
 * location parameter `mu` and scale parameter `sd`
 * 
 * @param mu location parameter of the Cauchy distribution
 * @param sd scale parameter of the Cauchy distribution
 * 
 * @return A random number from a Cauchy distribution with location `mu` and scale `sd`.
 */
double rcauchy(double mu, double sd) {
   unsigned long u, m = ~(1 << 31);
   u = rand();
   u &= m;
   return qcauchy(ldexp((double) u, -31), mu, sd);
}

/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dcauchy(x, 0.0, 1.0);
    p = pcauchy(x, 0.0, 1.0);
    q = qcauchy(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a Cauchy variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rcauchy(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
