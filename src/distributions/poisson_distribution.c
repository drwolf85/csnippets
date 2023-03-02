#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * The function calculates the probability mass function of a Poisson distribution for a given value of
 * x and lambda.
 * 
 * @param x The parameter "x" represents the number of events that we are interested in. It is an
 * integer value.
 * @param lambda The parameter "lambda" represents the average rate or intensity of the Poisson
 * distribution. It is used to calculate the probability of a certain number of events occurring in a
 * fixed interval of time or space.
 * 
 * @return the value of exp(z), where z is calculated based on the input values of x and lambda.
 */
double dpoisson(int x, double lambda) {
    int i;
    double z = nan("");
    if (x >= 0) if (lambda == 0.0) {
        return (double) (x == 0);
    }
    if (lambda > 0.0) {
        z = log(lambda) * (double) x;
        z -= lambda; 
        z -= lgamma((double) x + 1.0);
        z = exp(z);
    }
    return z;
}

/**
 * The function calculates the cumulative probability of a Poisson distribution up to a given value.
 * 
 * @param x The parameter "x" represents the number of events that we are interested in calculating the
 * probability for in a Poisson distribution.
 * @param lambda Lambda is the parameter of the Poisson distribution, which represents the average rate
 * at which events occur. It is a non-negative real number.
 * 
 * @return the sum of the probability mass function (PMF) values of the Poisson distribution for the
 * values 0 to x, given a specific lambda value.
 */
double ppoisson(int x, double lambda) {
    int i;
    double tmp, z = nan("");
    if (x >= 0 && lambda >= 0.0) {
        z = dpoisson(0, lambda);
        for (i = 1; i <= x; i++) {
            z += dpoisson(i, lambda);
        }
    }
    return z;
}

/**
 * The function qpoisson calculates the quantile of a Poisson distribution given a probability and a
 * lambda value.
 * 
 * @param p The parameter "p" represents the probability value for which we want to find the
 * corresponding quantile in the Poisson distribution.
 * @param lambda The parameter "lambda" in the qpoisson function represents the average rate or mean of
 * the Poisson distribution. It determines the shape and spread of the distribution.
 * 
 * @return a double value, which represents the quantile of the Poisson distribution.
 */
double qpoisson(double p, double lambda) {
    size_t i;
    double tmp, z = nan("");
    if (p >= 0.0 && p <= 1.0 && lambda >= 0.0) {
        z = dpoisson(0, lambda);
        for (i = 1; z <= p; i++) {
            z += dpoisson(i, lambda);
        }
        z = (double) (i - 1);    
    }
    return z;
}

/**
 * The function `rpoisson` generates a random number from a Poisson distribution with a given lambda
 * value.
 * 
 * @param lambda The parameter "lambda" represents the average rate at which events occur in a Poisson
 * distribution. It is a non-negative real number.
 * 
 * @return a double value, which is the result of the calculation performed inside the function.
 */
double rpoisson(double lambda) {
    unsigned long u, m;
    double z = nan("");
    if (lambda >= 0.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = qpoisson(ldexp((double) u, -31), lambda);
    }
    return z;
}

/* Test function */
int main() {
    double tmp;
    srand(time(NULL));
    printf("Test dpoission(x = 2, lambda = 1.5) = %f\n", dpoisson(2, 1.5));
    printf("Test ppoission(x = 5, lambda = 1.5) = %f\n", ppoisson(5, 1.5));
    printf("Test qpoission(x=0.9, lambda = 1.5) = %f\n", qpoisson(0.9, 1.5));
    printf("Test rpoission(lambda = 1.5) = %f\n", rpoisson(1.5));   
    /* Main function to test the random generation of a Poisson variable */
    for (int i = 1; i <= 144; i++) {
        tmp = rpoisson(1.685);
        if (tmp >= 0.0) printf(" ");
        printf("%1.f\t", tmp);
        if (i % 8 == 0) printf("\n");
    }
    return 0;
}
