#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the probability density of a log-normal distribution at a given value.
 * 
 * @param x The parameter "x" represents the value at which you want to evaluate the logarithmic normal
 * distribution.
 * @param m The parameter "m" represents the mean of the logarithm of the random variable.
 * @param s The parameter "s" in the above code represents the standard deviation of the logarithm of
 * the random variable.
 * 
 * @return the value of `z`, which is a double.
 */
double dlognorm(double x, double m, double s) {
    double z = 0.0;
    if (x > 0.0) {
        z = log(x) - m;
        z /= s; 
        z = exp(-0.5 * z * z);
        z /= x * s * sqrt(2.0 * M_PI);
    }
    return z;
}

/**
 * The function calculates the cumulative distribution function (CDF) of a log-normal distribution.
 * 
 * @param x The parameter "x" represents the value at which you want to evaluate the probability
 * density function (PDF) of the log-normal distribution.
 * @param m The parameter "m" represents the mean of the logarithm of the random variable.
 * @param s The parameter "s" in the above code represents the standard deviation of the logarithm of
 * the random variable.
 * 
 * @return the value of `z`, which is a double.
 */
double plognorm(double x, double m, double s) {
    double z = 0.0;
    if (x > 0.0) {
        z = log(x) - m;
        z /= s * sqrt(2.0); 
        z = 0.5 + 0.5 * erf(z);
    }
    return z;
}

/**
 * The function qlognorm calculates the quantile of a log-normal distribution given a probability,
 * mean, and standard deviation.
 * 
 * @param p The parameter "p" represents the probability value for which we want to find the
 * corresponding quantile. It should be a value between 0 and 1.
 * @param m The parameter "m" represents the location parameter of the log-normal distribution. It
 * determines the location of the peak of the distribution.
 * @param s The parameter "s" represents the scale parameter of the log-normal distribution. It
 * determines the spread or width of the distribution.
 * 
 * @return a double value, which is the quantile of the log-normal distribution with parameters m
 * (mean) and s (standard deviation) corresponding to the given probability p.
 */
double qlognorm(double p, double m, double s) {
    double old;
    double z = 0.25 * log(p / (1.0 - p)); /* Initial approximation */
    double sdv = dlognorm(exp(z), 0.0, 1.0);
    do {
        old = z;
        z += (p - plognorm(z, 0.0, 1.0)) / sdv;
        sdv = dlognorm(z, 0.0, 1.0);
    } while (sdv > 1e-9 && fabs(old - z) > 1e-12);
    return s * z + m;
}

/**
 * The function rlognorm generates a random number from a log-normal distribution with given mean (mu)
 * and standard deviation (sd).
 * 
 * @param mu The parameter "mu" represents the mean of the logarithm of the random variable you want to
 * generate.
 * @param sd The parameter "sd" in the above code refers to the standard deviation of the log-normal
 * distribution.
 * 
 * @return a random number generated from a log-normal distribution with the specified mean (mu) and
 * standard deviation (sd).
 */
double rlognorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = arc4random();
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
   return exp(mu + sd * a);
}

/* Test function */
int main() {
    double x = 1.64;
    double d, p, q;
    double tmp;
    d = dlognorm(x, 0.0, 1.0);
    p = plognorm(x, 0.0, 1.0);
    q = qlognorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a lognormal variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rlognorm(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
