#include <stdio.h>
#include <math.h>

double factorial(double x) { 
    double res = lgamma(x + 1.0);
    return exp(res);
}
