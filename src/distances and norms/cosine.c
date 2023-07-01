#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40

/**
 * The function calculates the cosine distance between two vectors of double values.
 * 
 * @param x A pointer to an array of double values representing the first vector.
 * @param y The parameter `y` is a pointer to a double array containing the second vector for which the
 * cosine distance is being calculated.
 * @param n The parameter "n" represents the size of the arrays "x" and "y", which are the input arrays
 * containing the vectors for which the cosine distance is being calculated.
 * 
 * @return the cosine distance between two vectors represented by arrays of double values.
 */
double cosine_distance(double *x, double *y, size_t n) {
    size_t i;
    double xn = 0.0, yn = 0.0, rs = 0.0;

    #pragma omp parallel for private(i) reduction(+ : xn, yn, rs)
    for (i = 0; i < n; i++) {
        xn += x[i] * x[i];
        yn += y[i] * y[i];
        rs += x[i] * y[i];
    }
    xn = sqrt(xn);
    xn *= sqrt(yn);
    xn = (xn > MIN_EPS) * xn + (xn <= MIN_EPS) * MIN_EPS;
    return sqrt(2.0 - 2.0 * rs / xn);
}

/* Test function */
int main() {
    double x[] = {1.0, 2.0, 0.5, -0.5, -2.0};
    double y[] = {-1.1, 1.9, 0.4, -0.4, 1.0};
    double cd = cosine_distance(x, y, 5);
    printf("Computed cosine distance is %f\n", cd);
    return 0;
}

