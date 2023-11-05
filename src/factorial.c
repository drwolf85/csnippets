#include <stdio.h>
#include <math.h>

double factorial(double x) { 
    double res = lgamma(x + 1.0);
    return exp(res);
}

double stirling(double x) {
    double res = x * log(x) - x;
    res += 0.5 * log(2.0 * M_PI * x); 
    return exp(res);
}

double gosper(double x) { 
    double res = x * log(x) - x;
    res += 0.5 * log(M_PI * (2.0 * x + 1.0 / 3.0)); 
    return exp(res);
}
