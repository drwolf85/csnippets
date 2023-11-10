#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the probability of a geometric distribution given the number of trials and
 * the success probability.
 * 
 * @param x The parameter "x" represents the number of failures before the first success in a geometric
 * distribution.
 * @param prob The parameter "prob" represents the probability of success in a geometric distribution.
 * 
 * @return the probability of observing the given number of failures (x) before the first success in a
 * geometric distribution with the given probability of success (prob).
 */
double dgeom(int x, double prob) {
    int i;
    double z = nan("");
    if (x >= 0 && prob >= 0.0 && prob <= 1.0) {
        z = prob;
        z *= pow(1.0 - prob, (double) x);
    }
    return z;
}

/**
 * The function calculates the probability of observing at least x successes in a geometric
 * distribution with probability of success prob.
 * 
 * @param x The parameter "x" represents the number of trials until the first success occurs in a
 * geometric distribution.
 * @param prob The parameter "prob" represents the probability of success in a geometric distribution.
 * 
 * @return the probability of observing the first success on the x-th trial in a geometric distribution
 * with probability of success prob.
 */
double pgeom(int x, double prob) {
    int i;
    double tmp, z = nan("");
    if (x >= 0 && prob >= 0.0 && prob <= 1.0) {
        z = 1.0 - pow(1.0 - prob, (double) x + 1.0);
    }
    return z;
}

/**
 * The function qgeom calculates the quantile of the geometric distribution given a probability and
 * success probability.
 * 
 * @param p The parameter "p" represents the probability of success in a geometric distribution. It
 * should be a value between 0 and 1, exclusive.
 * @param prob The parameter "prob" represents the probability of success in a geometric distribution.
 * 
 * @return the value of the variable "z".
 */
double qgeom(double p, double prob) {
    double z = nan("");
    if (p > 0.0 && p < 1.0 && prob > 0.0 && prob < 1.0) {
        z = log(1.0 - p) / log(1.0 - prob);
        z = ceil(z) - 1.0;    
    }
    else if ((p == 0.0 && prob == 0.0) || (p == 1.0 && prob == 1.0)) {
        z = 0.0;
    }
    else if (prob == 0.0 && p == 1.0) z = INFINITY;
    return z;
}

/**
 * The function `rgeom` generates a random number from a geometric distribution with a given
 * probability.
 * 
 * @param prob The parameter "prob" represents the probability of success in a geometric distribution.
 * It should be a value between 0 and 1, inclusive.
 * 
 * @return a double value.
 */
double rgeom(double prob) {
    unsigned long u, m;
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = qgeom(ldexp((double) u, -31), prob);
    }
    return z;
}

/* Test function */
int main() {
    int x = 2;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dgeom(x, 0.25);
    p = pgeom(x, 0.75);
    q = qgeom(0.95, 0.777);
    printf("x = %d, d = %f, p = %f, q = %.0f\n", x, d, p, q);
    /* Main function to test the random generation of a Geometric variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rgeom(0.678);
        printf("%.0f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
