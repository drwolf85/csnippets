#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40

/**
 * The function calculates the Pearson's distance between two vectors of double values.
 * 
 * @param x A pointer to an array of double values representing the first vector.
 * @param y The parameter `y` is a pointer to a double array containing the second vector for which the
 * Pearson's distance is being calculated.
 * @param n The parameter "n" represents the size of the arrays "x" and "y", which are the input arrays
 * containing the vectors for which the Pearson's distance is being calculated.
 * 
 * @return the Pearson's distance (based on Pearson's correlation) between two vectors represented by arrays of double values.
 */
double pearson_distance(double *x, double *y, size_t n) {
    size_t i;
    double xn = 0.0, yn = 0.0, rs = 0.0, mx = 0.0, my = 0.0;

    #pragma omp parallel for private(i) reduction(+ : xn, yn, rs, mx, my)
    for (i = 0; i < n; i++) {
        xn += x[i] * x[i];
        yn += y[i] * y[i];
        rs += x[i] * y[i];
        mx += x[i];
        my += y[i];
    }
    rs = rs * (double) n - mx * my;
    mx = sqrt((xn * (double) n - mx * mx) * (yn * (double) n - my * my));
    rs /= (mx > MIN_EPS) * mx + (mx <= MIN_EPS) * MIN_EPS;
    return 1.0 - rs;
}

/* Test function */
int main() {
    double x[] = {1.0, 2.0, 0.5, -0.5, -2.0};
    double y[] = {-1.1, 1.9, 0.4, -0.4, 1.0};
    double cd = pearson_distance(x, y, 5);
    printf("Computed Pearson's distance is %f\n", cd);
    return 0;
}

