#include <stdio.h>
#include <math.h>

double lbeta(double a, double b) {
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
    printf("lbeta(5, 2) = %f\n", lbeta(5,2));
    printf("beta(5, 2) = %f\n", beta(5,2));
    return 0;
}
