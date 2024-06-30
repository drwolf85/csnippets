#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


/**
 * The function calculates the Minkowski distance between two vectors of doubles with a given value of
 * p.
 * 
 * @param x A pointer to an array of double values representing the first vector.
 * @param y The parameter `y` is a pointer to an array of `double` values representing the coordinates
 * of a point in an n-dimensional space.
 * @param n `n` is the number of elements in the arrays `x` and `y`. It represents the dimensionality
 * of the space in which the Minkowski distance is being calculated.
 * @param p The parameter `p` is a positive real number that determines the order of the Minkowski
 * distance. When `p=1`, the distance is the Manhattan distance, and when `p=2`, the distance is the
 * Euclidean distance. For other values of `p`, the distance is known
 * 
 * @return The function `minkovsky_distance` returns a double value which represents the Minkovsky
 * distance between two vectors `x` and `y` of size `n` for a given value of `p`. If `p` is infinite,
 * then it computes Chebyshev's distance (as a limite case).
 */
double minkovsky_distance(double *x, double *y, size_t n, double p) {
    double res = 0.0;
    double tmp;
    size_t i = 0;
    if (isinf(p)) { /* If `p` is infinite, then compute Chebyshev's distance */
        tmp = fabs(x[i] - y[i]);
        #pragma omp parallel for simd reduction(max : res)
        for (i = 1; i < n; i++) {
            tmp = fabs(x[i] - y[i]);
            res = tmp > res ? tmp : res;
        }
    } 
    else { /* Otherwise, compute the Minkovsky distance for a give value of `p` */
        #pragma omp parallel for simd private(tmp) reduction(+ : res)
        for (i = 0; i < n; i++) {
            tmp = fabs(x[i] - y[i]);
            res += pow(tmp, p);
        }
        res = pow(res, 1.0 / p);
    } 
    return res;
}

/* Test function */
int main() {
    double x[] = {-5.2, 1.2, 4.6, 7.8, 9.8};
    double y[] = {1.2, 1.2, -3.4, 2.8, 9.6};
    double z[] = {0};
    double p, res;

    p = 1.0;
    res = minkovsky_distance(x, y, 5, p);
    printf("L-1 distance between x and y is %f\n", res);
    res /= 0.5 * (res + minkovsky_distance(x, z, 5, p) + minkovsky_distance(z, y, 5, p));
    printf("Steinhaus transform of L-1 distance between x and y is %f\n\n", res);

    p = 2.0;
    res = minkovsky_distance(x, y, 5, p);
    printf("L-2 distance between x and y is %f\n", res);
    res /= 0.5 * (res + minkovsky_distance(x, z, 5, p) + minkovsky_distance(z, y, 5, p));
    printf("Steinhaus transform of L-2 distance between x and y is %f\n\n", res);

    p = 3.0;
    res = minkovsky_distance(x, y, 5, p);
    printf("L-3 distance between x and y is %f\n", res);
    res /= 0.5 * (res + minkovsky_distance(x, z, 5, p) + minkovsky_distance(z, y, 5, p));
    printf("Steinhaus transform of L-3 distance between x and y is %f\n\n", res);

    p = INFINITY;
    res = minkovsky_distance(x, y, 5, p);
    printf("L-inf distance between x and y is %f\n", res);
    res /= 0.5 * (res + minkovsky_distance(x, z, 5, p) + minkovsky_distance(z, y, 5, p));
    printf("Steinhaus transform of L-inf distance between x and y is %f\n", res);
    return 0;
}
