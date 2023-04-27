#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the probability mass function of a Bernoulli distribution for a given value
 * and probability.
 * 
 * @param x The parameter x represents the outcome of a Bernoulli trial, which can only take on the
 * values 0 or 1.
 * @param prob The probability of success in a Bernoulli trial, where success is defined as the event
 * of interest (e.g. flipping heads on a coin).
 * 
 * @return The function `dbern` returns the probability mass function (PMF) value of a Bernoulli
 * distribution at a given value `x` and probability `prob`. If `prob` is not between 0 and 1, the
 * function returns `nan("")`, which stands for "not a number".
 */
double dbern(int x, double prob) {
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0) {
        z = (double) (x == 0) * (1 - prob);
        z += (double) (x == 1) * prob;
    }
    return z;
}

/**
 * The function calculates the cumulative probability mass function of a Bernoulli distribution for
 * a given value of x and probability parameter.
 * 
 * @param x The parameter x is an integer representing the outcome of a Bernoulli trial, where x can
 * only take the values 0 or 1.
 * @param prob The probability of success in a Bernoulli trial.
 * 
 * @return The function `pbern` returns a double value `z`. If the input `prob` is between 0 and 1, it
 * calculates the cumulative probability of getting a value of `x` using the Bernoulli distribution
 * formula and returns it. If `prob` is outside the range of 0 to 1, `z` remains `nan` (not a number).
 */
double pbern(int x, double prob) {
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0) {
        z = (double) (x == 0) * (1 - prob);
        z += (double) (x == 1);
    }
    return z;
}

/**
 * The function qbern returns a value of 1 if a given probability is greater than or equal to a
 * specified threshold, and 0 otherwise.
 * 
 * @param p The probability of a success in a Bernoulli trial.
 * @param prob The probability of success in a Bernoulli trial, where 0 <= prob <= 1.
 * 
 * @return The function `qbern` returns a `double` value, which is either 0.0 or 1.0, depending on
 * whether the input value `p` is less than or equal to the input value `prob`. If the input values are
 * outside the valid range, the function returns `nan("")`, which stands for "not a number".
 */
double qbern(double p, double prob) {
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0 && p >= 0.0 && p <= 1.0) {
        z = (double) (p <= prob);
    }
    return z;
}

/**
 * The function generates a random number between 0 and 1 and returns 1 if it is less than or equal to
 * a given probability, and 0 otherwise.
 * 
 * @param prob The parameter "prob" is a double value representing the probability of success in a
 * Bernoulli trial. It should be a value between 0 and 1, inclusive. The function uses this probability
 * to generate a random binary outcome (0 or 1) according to the Bernoulli distribution.
 * 
 * @return The function `rbern` returns a random variable that follows a Bernoulli distribution with
 * probability of success `prob`. It returns 1 with probability `prob` and 0 with probability `1-prob`.
 */
double rbern(double prob) {
    unsigned long u, m;
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = (double) (ldexp((double) u, -31) <= prob);        
    } 
    return z;
}

/* Test function */
int main() {
    double x = 1.0;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dbern(x, 0.5);
    p = pbern(x, 0.75);
    q = qbern(0.95, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a Bernoulli variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rbern(0.5);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
