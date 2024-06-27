#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function `dtriang` returns the value of the triangular density function at `x` 
 * with location parameter `m` and scale parameter `s` 
 * 
 * @param x the value we're evaluating the PDF at
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dtriang(double x, double m, double s) {
    double z = nan(""); 
    if (s >= 0.0) {
        z = x - m;
        s = 1.0 / s;
    }
    z *= s;
    z = (z >= -1.0) * (z <= 1.0) * (1.0 - fabs(z));
    return z * s;
}

/**
 * The function `ptriang` returns the probability that a random variable from a triangular distribution
 * with location parameter `m` and scale parameter `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double ptriang(double x, double m, double s) {
    double z = nan("");
    x -= m;
    if (s >= 0.0) {
        x /= s;
        z = 1.0 - fabs(x);
    }
    z *= (z > 0.0) * z * 0.5;
    z += (x >= 0) * (1.0 - 2.0 * z);
    return z;
}

/**
 * It takes a probability, a location, and a scale parameter, and returns the value of the triangular
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The quantile function of the triangular distribution.
 */
double qtriang(double p, double m, double s) {
    double z = nan("");
    double v = (p >= 0.5) * (1.0 - 2.0 * p);
    double sgn = 2.0 * (double) (p > 0.5) - 1.0;
    if (p >= 0.0 && p <= 1.0 && s >= 0.0) 
        z = p + v;
    z = sgn * (1.0 - sqrt(2.0 * z));
    return s * z + m;
}

/** 
 * The function rtriang() is a C function that generates a random number from a triangular distribution with
 * location parameter `mu` and scale parameter `sd`
 * 
 * @param mu location parameter of the triangular distribution
 * @param sd scale parameter of the triangular distribution
 * 
 * @return A random number from a triangular distribution with location `mu` and scale `sd`.
 */
double rtriang(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -31);
   b = ldexp((double) v, -31);
   return (a - b) * sd + mu;
}

/* Test function */
int main() {
    double x = 0.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dtriang(x, 0.0, 1.0);
    p = ptriang(x, 0.0, 1.0);
    q = qtriang(0.9352, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a triangular variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rtriang(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
