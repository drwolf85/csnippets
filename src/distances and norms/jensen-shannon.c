#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N_SPLITS 100

/**
 * The function calculates the Jensen-Shannon distance between two probability distributions
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
 * @return The function `jensen_shannon_distance` is returning the Jensen-Shannon distance between two
 * probability density functions represented by the input function pointers `x` and `y`.
 */
double jensen_shannon_distance(double (*x)(double), double (*y)(double)) {
    double res = 0.0;
    double z, f, g, cm, tmp, s;
    double const inv = 1.0 / (double) N_SPLITS;
    size_t i = 0;

    #pragma omp parallel for simd private(z, f, g, cm, tmp, s) reduction(+ : res)
    for (i = 0; i < N_SPLITS; i++) {
        cm = 0.0;
        z = (double) i * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        s = 2.0 / (f + g);
        tmp = f * log(f * s);
        if (isfinite(tmp)) cm += tmp;
        tmp = g * log(g * s);
        if (isfinite(tmp)) cm += tmp;
        z = (double) (i + 1) * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        s = 2.0 / (f + g);
        tmp = f * log(f * s);
        if (isfinite(tmp)) cm += tmp;
        tmp = g * log(g * s);
        if (isfinite(tmp)) cm += tmp;
        /* Linear approximation */ /*
        res += 0.5 * cm * inv;
        continue;
        /* Simpson's rule */
        z = ((double) i + 0.5) * inv;
        /* z = tan(M_PI * (z - 0.5)); */
        f = (*x)(z);
        g = (*y)(z);
        s = 2.0 / (f + g);
        tmp = f * log(f * s);
        if (isfinite(tmp)) cm += 4.0 * tmp;
        tmp = g * log(g * s);
        if (isfinite(tmp)) cm += 4.0 * tmp;
        res += cm * inv / 6.0;
    }
    return sqrt(fabs(res));
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
    printf("JS-dist unif-vs-unif = %f\n", jensen_shannon_distance(dunif, dunif));
    printf("JS-dist quad-vs-quad = %f\n", jensen_shannon_distance(dquad, dquad));
    printf("JS-dist unif-vs-quad = %f\n", jensen_shannon_distance(dunif, dquad));
    printf("JS-dist quad-vs-unif = %f\n", jensen_shannon_distance(dquad, dunif));
    return 0;
}
