#include <stdio.h>
#include <math.h>

double inline lbeta(double a, double b) {
    double res = lgamma(a);
    res += lgamma(b);
    res -= lgamma(a + b);
    return res;
}

double beta(double a, double b) {
    double res = lbeta(a, b);
    return exp(res);
}

/* Test function */
int main() {
    printf("beta(5, 2) = %f\n", beta(5,2));
    return 0;
}
