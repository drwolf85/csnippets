#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * `dunif` returns the probability density of a uniform 
 * distribution with lower bound `a` and upper bound `b` at `x`
 * 
 * @param x the value to be evaluated
 * @param a lower bound
 * @param b the upper bound of the uniform distribution
 * 
 * @return The probability of x being between a and b.
 */
double dunif(double x, double a, double b) {
    double z = nan("");
    if (b > a) {
        z = (double) (x >= a & x <= b) / (b - a);
    }
    return z;
}

/**
 * `punif` returns the probability that a uniform random variable is
 * less than or equal to `x` given that it is between `a` and `b`
 * 
 * @param x the value to be evaluated
 * @param a lower limit of the uniform distribution
 * @param b the upper bound of the uniform distribution
 * 
 * @return the probability of a uniform distribution.
 */
double punif(double x, double a, double b) {
    double z = nan("");
    if (b > a) {
        z = (double) (x > a) * (x - a) / (b - a);
        z *= (double) (x < b);
        z += (double) (x >= b);
    }
    return z;
}

/**
 * `qunif` returns a quantile of a uniform distribution 
 * with minimum `a` and maximum `b` for a given probability `p`
 * 
 * @param p the probability of the random variable being less than or equal to the value
 * @param a lower bound
 * @param b upper bound
 * 
 * @return A double
 */
double qunif(double p, double a, double b) {
    double z = nan("");
    if (b > a && p >= 0.0 && p <= 1.0) {
        z = p * (b - a) + a;
    }
    return z;
}

/**
 * `runif` generates a random number between `a` and `b` using the `rand` function
 * 
 * @param a lower bound
 * @param b upper bound
 * 
 * @return A random number between a and b.
 */
double runif(double a, double b) {
    unsigned long u, m = ~(1 << 31);
    if (b > a) {
        u = rand() & m;
        return ldexp((double) u, -31) * (b - a) + a;
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
    d = dunif(x, 0.0, 2.0);
    p = punif(x, 0.0, 2.0);
    q = qunif(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a uniform variable */
    for (int i = 1; i <= 40; i++) {
        tmp = runif(-1.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
