#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the binomial coefficient using the gamma function.
 * 
 * @param n The total number of items in the set or population.
 * @param k k represents the number of objects to be chosen from a set of n objects in a binomial
 * distribution.
 * 
 * @return The function `binom` returns the value of the binomial coefficient "n choose k", which
 * represents the number of ways to choose k items from a set of n distinct items, without regard to
 * order. The function calculates this value using the formula n! / (k! * (n-k)!), but uses logarithmic
 * and exponential functions to avoid numerical overflow or underflow.
 */
double binom(int n, int k) {
    double res = lgamma((double) (n + 1));
    res -= lgamma((double) (k + 1));
    res -= lgamma((double) (n - k + 1)); 
    return exp(res);
}

/**
 * The function calculates the probability of getting x successes in n independent trials with a given
 * probability of success.
 * 
 * @param x The number of successes in the binomial distribution.
 * @param n The parameter "n" represents the total number of trials in a binomial experiment.
 * @param prob The probability of success in a single trial of a binomial experiment.
 * 
 * @return a double value, which is the probability of getting exactly x successes in n independent
 * Bernoulli trials with probability of success prob.
 */
double dbinom(int x, int n, double prob) {
    int i;
    double tmp = 0.0, z = nan("");
    if (prob >= 0.0 && prob <= 1.0 && n >= 0) {
        z = binom(n, x);
        z *= pow(prob, (double) x);
        z *= pow(1.0 - prob, (double) (n - x));
    }
    return z;
}

/**
 * The function calculates the probability of getting x or fewer successes in n independent Bernoulli
 * trials with probability of success prob.
 * 
 * @param x The number of successes in the binomial distribution.
 * @param n The parameter "n" represents the total number of trials in a binomial distribution.
 * @param prob The probability of success in a single trial of a binomial experiment.
 * 
 * @return a double value, which is the probability of getting x or fewer successes in a binomial
 * distribution with n trials and a given probability of success (prob).
 */
double pbinom(int x, int n, double prob) {
    int i;
    double tmp, z = nan("");
    if (x >= 0 && x <= n && prob >= 0.0 && prob <= 1.0 && n >= 0) {
        z = dbinom(0, n, prob);
        for (i = 1; i <= x; i++) {
            z += dbinom(i, n, prob);
        }
    }
    return z;
}

/**
 * The function qbinom calculates the inverse of the cumulative distribution function of a binomial
 * distribution.
 * 
 * @param p The probability threshold for the quantile function. The function will return the smallest
 * integer i such that the cumulative probability of the binomial distribution is less than p.
 * @param n The parameter "n" represents the number of trials in a binomial distribution.
 * @param prob The probability of success in a Bernoulli trial.
 * 
 * @return a double value, which is the number of successes (i.e., the number of times an event of
 * interest occurs) in a Bernoulli trial with a given probability of success and a given number of
 * trials, such that the probability of observing the returned number of successes or fewer is less
 * than or equal to a given probability value.
 */
double qbinom(double p, int n, double prob) {
    int i;
    double tmp, z = nan("");
    if (p >= 0.0 && p <= 1.0 && prob >= 0.0 && prob <= 1.0 && n >= 0) {
        z = dbinom(0, n, prob);
        for (i = 1; i <= n; i++) {
            z += dbinom(i, n, prob);
            if (z > p) {
                break;
            }
        }
        z = (double) (i - 1);    
    }
    return z;
}

/**
 * The function generates a random number of successes in a given number of trials based on a given
 * probability.
 * 
 * @param n The parameter "n" represents the number of trials in the binomial distribution.
 * @param prob The probability of success for each trial in the binomial distribution.
 * 
 * @return a double value, which represents the number of successes in n independent Bernoulli trials
 * with probability of success equal to prob.
 */
double rbinom(int n, double prob) {
    int i;
    unsigned long u, m;
    double z = nan("");
    if (n >= 0 && prob >= 0.0 && prob <= 1.0) {
        z = 0.0;
        for (i = 0; i < n; i++) {
            u = rand();
            m = ~(1 << 31);
            u &= m;
            z += (double) (ldexp((double) u, -31) <= prob);        
        }
    }
    return z;
}

/* Test function */
int main() {
    double x = 1.0;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dbinom(x, 5, 0.25);
    p = pbinom(x, 5, 0.75);
    q = qbinom(0.95, 5, 0.777);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a Binomial variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rbinom(5, 0.678);
        if (tmp >= 0.0) printf(" ");
        printf("%1.f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
