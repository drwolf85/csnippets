#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * τ=2(nc−nd) / n(n−1).
 * 
 */
#define MIN_EPS 1e-40
#define SIGN(x, y) (((x) > (y)) - ((x) < (y)))
/**
 * The function calculates the Kendall's distance between two vectors of double values.
 * 
 * @param x A pointer to an array of double values representing the first vector.
 * @param y The parameter `y` is a pointer to a double array containing the second vector for which the
 * Kendall's distance is being calculated.
 * @param n The parameter "n" represents the size of the arrays "x" and "y", which are the input arrays
 * containing the vectors for which the Kendall's distance is being calculated.
 * 
 * @return the Kendall's distance (based on Kendall's tau) between two vectors represented by arrays of double values.
 */
double kendall_distance(double *x, double *y, size_t n) {
    size_t i, j;
    long long sm = 0;
    double tau;
    #pragma omp parallel for private(i, j) reduction(+ : sm) collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sm += SIGN(x[i], x[j]) * SIGN(y[i], y[j]);
        }
    }
    tau = 2.0 * (double) sm / (double) (n * (n - 1));
    return 1.0 - tau;
}

/* Test function */
int main() {
    double x[] = {1.0, 2.0, 0.5, -0.5, -2.0};
    double y[] = {-1.1, 1.9, 0.4, -0.4, 1.0};
    double cd = kendall_distance(x, y, 5);
    printf("Computed Kendall's distance is %f\n", cd);
    return 0;
}

