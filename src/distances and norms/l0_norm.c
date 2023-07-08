#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-15

/**
 * The function calculates the l0 norm of a given array by counting the number of non-zero elements.
 * 
 * @param x An array of double values representing the input vector.
 * @param n The parameter `n` represents the size of the array `x`.
 * 
 * @return The function `l0_norm` returns the sum of the absolute values of the elements in the array
 * `x` that are greater than `MIN_EPS`.
 */
double l0_norm(double *x, int n) {
    int i;
    double res = 0.0;
    #pragma omp parallel for simd reduction (+ : res)
    for (i = 0; i < n; i++) {
        res += (double) (fabs(x[i]) > MIN_EPS);
    }
    return res;
}

/* Test function */
int main() {
    double vec[] = {0.1, -0.1, 0.5};
    printf("L0-Norm of `vec` is %.1f\n", 
           l0_norm(vec, 3));
    return 0;
}
