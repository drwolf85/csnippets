#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define CLAMP(XX, LL, UU) (((XX) <= (LL)) * (LL) + ((XX) >= (UU)) * (UU) + (((XX) > (LL)) && ((XX) < (UU))) * (XX))
#define MIN(XX, YY) (((XX) <= (YY)) * (XX) + ((XX) > (YY)) * (YY))
#define MAX(XX, YY) (((XX) >= (YY)) * (XX) + ((XX) < (YY)) * (YY))

static inline double fuzzines(double x) {
    return MIN(x, 1.0 - x);
}

static inline bool consonant(double x, double y) {
    return (MAX(x, y) <= 0.5) || (MIN(x, y) >= 0.5);
}

/**
 * The function calculates the Muljačić distance between two arrays of numbers using parallel processing.
 * 
 * @param x The parameter `x` is a pointer to an array representing the first vector for which
 * we want to calculate the Muljačić distance.
 * @param y The parameter `y` is a pointer to another array, which is used as input to the
 * `fuzzy_hamming_distance` function to compute its distance from `x`.
 * @param n The parameter `n` represents the length of the input arrays `x` and `y`.
 * 
 * @return The function `fuzzy_hamming_distance` returns the Muljačić distance between two strings `x` and `y`.
 */
double fuzzy_hamming_distance(double *x, double *y, size_t n) {
    size_t i;
    bool tmp;
    double fzx, fzy;
    double res = 0;
    double *xv = (double *) malloc(n * sizeof(double));
    double *yv = (double *) malloc(n * sizeof(double));
    if (xv && yv) {
        /* Preprocess input arrays */
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            xv[i] = fabs(CLAMP(x[i], 0.0, 1.0));
            yv[i] = fabs(CLAMP(y[i], 0.0, 1.0));
        }
        #pragma omp parallel for private(i, tmp, fzx, fzy) reduction(+ : res)
        for (i = 0; i < n; i++) {
            tmp = consonant(xv[i], yv[i]);
            fzx = fuzzines(xv[i]);
            fzy = fuzzines(yv[i]);
            fzx = MAX(fzx, fzy);
            res += ((double) !tmp) + (2.0 * (double) tmp - 1.0) * fzx;
        }
    }
    free(xv);
    free(yv);
    return res;
}

/* Test function */
int main() {
    double x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    double y[] = {0.5, 0.3, 0.6, 0.1, 0.4, 0.7, 0.2};
    double z[] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    double hd = fuzzy_hamming_distance(x, y, 7);
    printf("Computed Muljačić distance is %f\n", hd);
    hd /= 0.5 * (hd + fuzzy_hamming_distance(x, z, 7) + fuzzy_hamming_distance(y, z, 7));
    printf("Computed Steinhaus transfrom with Muljačić distance, and it is %f\n", hd);
    return 0;
}
