#include <stdio.h>
#include <math.h>

double dnorm(double x, double m, double s) {
    double z = x - m;
    z /= s; 
    z = exp(-0.5 * z * z);
    z /= s * sqrt(2.0 * M_PI);
    return z;
}

double pnorm(double x, double m, double s) {
    double z = x - m;
    z /= s * sqrt(2.0); 
    z = 0.5 + 0.5 * erf(z);
    return z;
}

double qnorm(double p, double m, double s) {
    double old;
    double z = 0.25 * log(p / (1.0 - p)); /* Initial approximation */
    double sdv = dnorm(z, 0.0, 1.0);
    do {
        old = z;
        z += (p - pnorm(z, 0.0, 1.0)) / sdv;
        sdv = dnorm(z, 0.0, 1.0);
    } while (sdv > 1e-9 && fabs(old - z) > 1e-12);
    return s * z + m;
}

int main() {
    double x = -1.64;
    double d, p, q;
    d = dnorm(x, 0.0, 1.0);
    p = pnorm(x, 0.0, 1.0);
    q = qnorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    return 0;
}

