#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double dgeom(int x, double prob) {
    int i;
    double z = nan("");
    if (x >= 0 && prob >= 0.0 && prob <= 1.0) {
        z = prob;
        z *= pow(1.0 - prob, (double) x);
    }
    return z;
}

double pgeom(int x, double prob) {
    int i;
    double tmp, z = nan("");
    if (x >= 0 && prob >= 0.0 && prob <= 1.0) {
        z = 1.0 - pow(1.0 - prob, (double) x + 1.0);
    }
    return z;
}

double qgeom(double p, double prob) {
    double z = nan("");
    if (p > 0.0 && p < 1.0 && prob > 0.0 && prob < 1.0) {
        z = log(1.0 - p) / log(1.0 - prob);
        z = ceil(z) - 1.0;    
    }
    else if ((p == 0.0 && prob == 0.0) || (p == 1.0 && prob == 1.0)) {
        z = 0.0;
    }
    else if (prob == 0.0 && p == 1.0) z = INFINITY;
    return z;
}

double rgeom(double prob) {
    unsigned long u, m;
    double z = nan("");
    if (prob >= 0.0 && prob <= 1.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = qgeom(ldexp((double) u, -31), prob);
    }
    return z;
}

/* Test function */
int main() {
    int x = 2;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dgeom(x, 0.25);
    p = pgeom(x, 0.75);
    q = qgeom(0.95, 0.777);
    printf("x = %d, d = %f, p = %f, q = %.0f\n", x, d, p, q);
    /* Main function to test the random generation of a Geometric variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rgeom(0.678);
        printf("%.0f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
