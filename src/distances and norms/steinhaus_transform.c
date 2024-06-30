/* Steinhaus transform */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N_SPLITS 100

/**
 * @brief  * The function calculates the Steinhaus transform between two arrays for a given pivot
 * point in the same metric space of the two arrays.
 * 
 * @param x pointer to the first array of data
 * @param y pointer to the second array of data
 * @param z pointer to the pivot array (usually kept constant)
 * @param n size of the arrays `x`, `y`, and `z`
 * @param vec_dist pointer to a distance function applied in (or defining) a vectorial metric space
 * 
 * @return The function `steinhaus_transform` is returning the Steinhaus transform between two
 * arrays represented by the input function pointers `x` and `y` using the array `z` as a pivot point.
 */
extern inline double steinhaus_transform(double *x, double *y, double *z, size_t n, double (*vec_dist)(double *, double *, size_t)) {
    double den = vec_dist(x, y, n);
    double num = 2.0 * den;
    den += vec_dist(x, z, n);
    den += vec_dist(y, z, n);
    return num / den;
}

/**
 * The function calculates the Steinhaus transform between two distribution-density functions using
 * a third density as a pivot distribution.
 * 
 * @param x The parameter `x` is a pointer to a function that takes a double as input and returns a
 * double as output. This function represents a probability density function (PDF) that we want to
 * compare with another PDF represented by the function `y`. The function `x` is evaluated `N_SPLITS` 
 * times and it is assumed to have support $[0, 1]$.
 * @param y The parameter `y` is a pointer to a function that takes a double argument and returns a
 * double value. This function represents a probability density function (PDF) that we want to compare
 * with another PDF represented by the function `x`. The same support of the two PDFs is required.
 * @param z The parameter `z` is a pointer to a function that takes a double argument and returns a
 * double value. This function represents a probability density function (PDF) that we want to use as
 * a pivot PDF. The same support of `x` and `y` is required.
 * @param distro_dist Pointer to a distance function that can take in input the pointers `x`, `y`, 
 * and `z`. 
 * 
 * @return The function `steinhaus_transform_for_distributions` is returning the Steinhaus transform
 * between two probability density functions represented by the input function pointers `x` and `y`
 * using the probability density function `z` as a pivot distribution.
 */
extern inline double steinhaus_transform_for_distributions(double (*x)(double), double (*y)(double), double (*z)(double), 
                                                           double (*distro_dist)(double (*)(double), double (*)(double))) {
    double den = distro_dist(x, y);
    double num = 2.0 * den;
    den += distro_dist(x, z);
    den += distro_dist(y, z);
    return num / den;

}

/* Test functions */
static double euclidean_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    double tmp;
    size_t i = 0;
    #pragma omp parallel for simd private(tmp) reduction(+ : res)
    for (i = 0; i < n; i++) {
        tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    res = sqrt(res);
    return res;
}

static double dunif(double x) {
    return 1.0;
}

static double dquad(double x) {
    return x * x * 3.0;
}

static double dpivot(double x) {
    return 3.0 / 2.0 * sqrt(x);
}

static double hellinger_distance(double (*x)(double), double (*y)(double)) {
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

int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6};
    double z[] = {0.5, 0.5, 0.5, 0.5, 0.5 };
    double res = steinhaus_transform(x, y, z, 5, euclidean_distance);
    printf("Steinhaus transform of Euclidean distances between x and y is %f\n", res);
    res = steinhaus_transform_for_distributions(dunif, dquad, dpivot, hellinger_distance);
    printf("Steinhaus transform of Hellinger distances between two distributions is %f\n", res);
    return 0;
}
