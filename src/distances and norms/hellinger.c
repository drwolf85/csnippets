#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N_SPLITS 100

/**
 * The function calculates the Hellinger distance between two probability distributions
 * represented by functions.
 * 
 * @param x The parameter `x` is a pointer to a function that takes a double as input and returns a
 * double as output. This function represents a probability density function (PDF) that we want to
 * compare with another PDF represented by the function `y`. The function `x` is evaluated `N_SPLITS` 
 * times and it is assumed to have support $[0, 1]$.
 * @param y The parameter `y` is a pointer to a function that takes a double argument and returns a
 * double value. This function represents a probability density function (PDF) that we want to compare
 * with another PDF represented by the function `x`. The same support of the two PDFs is required.
 * 
 * @return The function `hellinger_distance` is returning the Hellinger distance between two
 * probability density functions represented by the input function pointers `x` and `y`.
 */
double hellinger_distance(double (*x)(double), double (*y)(double)) {
    double res = 0.0;
    double z, f, g, cm, tmp;
    double const inv = 1.0 / (double) N_SPLITS;
    size_t i = 0;

    #pragma omp parallel for simd private(z, f, g, cm, tmp) reduction(+ : res)
    for (i = 0; i < N_SPLITS; i++) {
        cm = 0.0;
        z = (double) i * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        tmp = sqrt(f * g);
        if (isfinite(tmp)) cm += tmp;
        z = (double) (i + 1) * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        tmp = sqrt(f * g);
        if (isfinite(tmp)) cm += tmp;
        /* Linear approximation */ /*
        res += 0.5 * cm * inv;
        continue;
        /* Simpson's rule */
        z = ((double) i + 0.5) * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        tmp = sqrt(f * g);
        if (isfinite(tmp)) cm += 4.0 * tmp;
        res += cm * inv / 6.0;
    }
    return sqrt(fabs(1.0 - res));
}

/* Test functions */
double dunif(double x) {
    return 1.0;
    /* double res = 0.0;
    res += (x >= 0.0 && x <= 1.0);
    return res; */
}

double dquad(double x) {
    return x * x * 3.0;
    /* double res = 0.0;
    res += (x >= 0.0 && x <= 1.0);
    return res * x * x * 3.0; */
}

int main () {
    printf("Hellinger-dist unif-vs-unif = %f\n", hellinger_distance(dunif, dunif));
    printf("Hellinger-dist quad-vs-quad = %f\n", hellinger_distance(dquad, dquad));
    printf("Hellinger-dist unif-vs-quad = %f\n", hellinger_distance(dunif, dquad));
    printf("Hellinger-dist quad-vs-unif = %f\n", hellinger_distance(dquad, dunif));
    return 0;
}
