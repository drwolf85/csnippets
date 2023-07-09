#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function `dcos` returns the value of the cosine density function at `x` 
 * with location parameter `m` and scale parameter `s` 
 * 
 * @param x the value we're evaluating the PDF at
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dcos(double x, double m, double s) {
    double z = x - m;
    s = 1.0 / s;
    z *= s;
    z = (z >= -M_PI) * (z <= M_PI) * (1.0 + cos(z)) * s;
    return z * (0.5 / M_PI);
}

/**
 * The function `pcos` returns the probability that a random variable from a cosine distribution
 * with location parameter `m` and scale parameter `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double pcos(double x, double m, double s) {
    double z;
    x -= m;
    x /= s;
    z = (M_PI + x + sin(x)) / (2.0 * M_PI);
    z = (x > -M_PI) * (x < M_PI) * z;
    z += (x >= M_PI);
    return z;
}

/**
 * It takes a probability, a location, and a scale parameter, and returns the value of the cosine
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The quantile function of the cosine distribution.
 */
double qcos(double p, double m, double s) {
    double sdv, old, z = nan("");
    if (p >= 0.0 && p <= 1.0 && s >= 0.0) 
        z = M_PI * (2.0 * p - 1.0);
    p = z;
    sdv = 1.0 + cos(z);
    do {
        old = z;
        z += (p - z - sin(z)) / sdv;
        sdv = 1.0 + cos(z);
    } while (sdv > 1e-9 && fabs(old - z) > 1e-12);
    return s * z + m;
}

/** 
 * The function rcos() is a C function that generates a random number from a cosine distribution with
 * location parameter `mu` and scale parameter `sd`
 * 
 * @param mu location parameter of the cosine distribution
 * @param sd scale parameter of the cosine distribution
 * 
 * @return A random number from a cosine distribution with location `mu` and scale `sd`.
 */
double rcos(double mu, double sd) {
   unsigned long u, m = ~(1 << 31);
   double a, b, s;
   u = rand();
   u &= m;
   return qcos(ldexp((double) u, -31), mu, sd);
}

/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dcos(x, 0.0, 1.0);
    p = pcos(x, 0.0, 1.0);
    q = qcos(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a cosine variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rcos(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
