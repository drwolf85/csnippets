#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double dpoisson(int x, double lambda) {
    int i;
    double z = nan("");
    if (x >= 0) if (lambda == 0.0) {
        return (double) (x == 0);
    }
    else if (lambda > 0.0) {
        z = log(lambda) * (double) x;
        z -= lambda; 
        z -= lgamma((double) x + 1.0);
    }
    return exp(z);
}

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
