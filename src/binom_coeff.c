#include <stdio.h>
#include <math.h>

/**
 * The function calculates the binomial coefficient of two integers using the logarithmic gamma
 * function.
 * 
 * @param n The parameter "n" represents the total number of items in a set from which we want to
 * choose "k" items.
 * @param k The parameter "k" in the function "binom_coefficient" represents the number of objects
 * chosen from a set of "n" distinct objects. It is used to calculate the binomial coefficient, which
 * is the number of ways to choose "k" objects from a set of "n" objects.
 * 
 * @return The function `binom_coefficient` returns the binomial coefficient of `n` choose `k`, which
 * is the number of ways to choose `k` items from a set of `n` items without regard to order. The
 * function calculates this value using the formula `n! / (k! * (n-k)!)`, but uses logarithmic and
 * exponential functions to avoid numerical issues. 
 */
double binom_coefficient(int n, int k) {
    double res = lgamma((double) (n + 1));
    res -= lgamma((double) (k + 1));
    res -= lgamma((double) (n - k + 1)); 
    return exp(res);
}

/* Test function */
int main() {
    printf("binom(5, 2) = %f\n", binom_coefficient(5,2));
    return 0;
}
