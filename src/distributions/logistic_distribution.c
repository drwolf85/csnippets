#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function `dlogis` returns the value of the logistic density function at `x` 
 * with location parameter `m` and scale parameter `s` 
 * 
 * @param x the value we're evaluating the PDF at
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dlogis(double x, double m, double s) {
    double z = m - x;
    s = 1.0 / s; 
    z = 1.0 / (1.0 + exp(z * s));
    return z * (1.0 - z) * s;
}

/**
 * The function `plogis` returns the probability that a random variable from a logistic distribution
 * with location parameter `m` and scale parameter `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double plogis(double x, double m, double s) {
    double z = m - x;
    z /= s; 
    z = 1.0 + exp(z);
    return 1.0 / z;
}

/**
 * It takes a probability, a location, and a scale parameter, and returns the value of the logistic
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m the location parameter
 * @param s the scale parameter
 * 
 * @return The quantile function of the logistic distribution.
 */
double qlogis(double p, double m, double s) {
    double z = log(p / (1.0 - p));
    return s * z + m;
}

/** 
 * The function rlogis() is a C function that generates a random number from a logistic distribution with
 * location parameter `mu` and scale parameter `sd`
 * 
 * @param mu location parameter of the logistic distribution
 * @param sd scale parameter of the logistic distribution
 * 
 * @return A random number from a logistic distribution with location `mu` and scale `sd`.
 */
double rlogis(double mu, double sd) {
   unsigned long u, m = ~(1 << 31);
   double a, b, s;
   u = rand();
   u &= m;
   return qlogis(ldexp((double) u, -31), mu, sd);
}

/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dlogis(x, 0.0, 1.0);
    p = plogis(x, 0.0, 1.0);
    q = qlogis(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a logistic variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rlogis(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
