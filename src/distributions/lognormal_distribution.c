#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double dlognorm(double x, double m, double s) {
    double z = 0.0;
    if (x > 0.0) {
        z = log(x) - m;
        z /= s; 
        z = exp(-0.5 * z * z);
        z /= x * s * sqrt(2.0 * M_PI);
    }
    return z;
}

double plognorm(double x, double m, double s) {
    double z = 0.0;
    if (x > 0.0) {
        z = log(x) - m;
        z /= s * sqrt(2.0); 
        z = 0.5 + 0.5 * erf(z);
    }
    return z;
}

double qlognorm(double p, double m, double s) {
    double old;
    double z = 0.25 * log(p / (1.0 - p)); /* Initial approximation */
    double sdv = dlognorm(exp(z), 0.0, 1.0);
    do {
        old = z;
        z += (p - plognorm(z, 0.0, 1.0)) / sdv;
        sdv = dlognorm(z, 0.0, 1.0);
    } while (sdv > 1e-9 && fabs(old - z) > 1e-12);
    return s * z + m;
}

double rlognorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -30) - 1.0;
   s = a * a;
   b = ldexp((double) v, -30) - 1.0;
   s += b * b * (1.0 - s);
   s = -2.0 * log(s) / s;
   a = b * sqrtf(s);
   return exp(mu + sd * a);
}

int main() {
    double x = 1.64;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dlognorm(x, 0.0, 1.0);
    p = plognorm(x, 0.0, 1.0);
    q = qlognorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a lognormal variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rlognorm(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
