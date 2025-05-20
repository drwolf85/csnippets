#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the logarithm of the binomial coefficient for given values of n and k.
 * 
 * @param n The parameter `n` represents the total number of items in the set.
 * @param k The parameter "k" represents the number of successes in a binomial distribution. It is the
 * number of events or objects we are interested in out of a total of "n" events or objects.
 * 
 * @return the logarithm of the binomial coefficient "n choose k".
 */
double log_binom(int n, int k) {
    double res = lgamma((double) (n + 1));
    res -= lgamma((double) (k + 1));
    res -= lgamma((double) (n - k + 1)); 
    return res;
}

/**
 * The function calculates the probability mass function of the hypergeometric distribution.
 * 
 * @param x The parameter "x" represents the number of white balls in the sample.
 * @param m The parameter "m" represents the number of white balls in the urn.
 * @param n The parameter "n" represents the number of black balls in the urn.
 * @param k The parameter "k" represents the number of balls drawn from the urn.
 * 
 * @return a double value, which is the result of the calculations performed in the function.
 */
double dhyper(int x, int m, int n, int k) {
    double z = nan("");
    if ((x >= 0 && x >= (k - n)) && (x <= k && x <= m)) {
        z = log_binom(m, x);
        z += log_binom(n, k - x);
        z -= log_binom(m + n, k);      
        z = exp(z);
    }
    return z;
}

/**
 * The function `phyper` calculates the hypergeometric distribution function for a given set of
 * parameters.
 * 
 * @param x The parameter "x" represents the number of white balls in the sample.
 * @param m The parameter "m" represents the number of white balls in the urn.
 * @param n The parameter "n" represents the number of black balls in the urn.
 * @param k The parameter "k" represents the number of balls drawn from the urn.
 * 
 * @return a double value.
 */
double phyper(int x, int m, int n, int k) {
    int j, i = k - n;
    double z = nan("");
    i *= (i > 0);
    if (x >= i && (x <= k && x <= m)) {
        z = 0.0;
        for (j = 0; j < x; j++) {
            z += dhyper(i + j, m, n, k);
        }
    }
    return z;
}

/**
 * The function qhyper calculates the quantile of the hypergeometric distribution.
 * 
 * @param p The parameter "p" represents the probability value for which we want to find the quantile.
 * It should be a value between 0 and 1.
 * @param m The parameter "m" represents the number of white balls in the urn.
 * @param n The parameter "n" represents the number of black balls in the urn.
 * @param k The parameter "k" represents the number of balls drawn from the urn.
 * 
 * @return a double value, which is the quantile of the hypergeometric distribution.
 */
double qhyper(double p, int m, int n, int k) {
    int j, i = k - n;
    double z = nan("");
    i *= (i > 0);
    if (p >= 0.0 && p < 1.0) {
        z = 0.0;
        for (j = 0; z < p; j++) {
            z += dhyper(i + j, m, n, k);
        }
        z = (double) (i + j - 1);
        if ((z > (double) k || z > (double) m)) z = nan("");
    }
    else if (p == 1.0) {
        z = (double) (k * (k < m) + m * (m >= k));
    }
    return z;
}

/**
 * The function `rhyper` generates a random number from a hypergeometric distribution.
 * 
 * @param m The parameter "m" represents the number of white balls in the urn.
 * @param n The parameter "n" represents the number of black balls in the urn.
 * @param k The parameter "k" represents the number of balls drawn from the urn.
 * 
 * @return a double value.
 */
double rhyper(int m, int n, int k) {
    unsigned long u, v;
    double z;
    u = arc4random();
    v = ~(1 << 31);
    u &= v;
    z = qhyper(ldexp((double) u, -31), m, n, k);
    return z;
}

/* Test function */
int main() {
    int x = 2;
    double d, p, q;
    double tmp;
    d = dhyper(x, 5, 2, 3);
    p = phyper(x, 5, 2, 3);
    q = qhyper(0.95, 5, 2, 3);
    printf("x = %d, d = %f, p = %f, q = %.0f\n", x, d, p, q);
    /* Main function to test the random generation of a Hypergeometric variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rhyper(5, 2, 3);
        printf("%.0f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
