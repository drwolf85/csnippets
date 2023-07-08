#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the L1 norm of a given array of doubles.
 * 
 * @param x The parameter `x` is a pointer to an array of `double` values. It represents the input
 * vector for which we want to calculate the L1 norm.
 * @param n The parameter `n` represents the size of the array `x`. It indicates the number of elements
 * in the array that need to be considered for calculating the L1 norm.
 * 
 * @return the L1 norm of the input array `x`.
 */
double l1_norm(double *x, int n) {
    int i;
    double res = 0.0;
    #pragma omp parallel for simd reduction (+ : res)
    for (i = 0; i < n; i++) {
        res += fabs(x[i]);
    }
    return res;
}

/* Test function */
int main() {
    double vec[] = {0.1, -0.1, 0.5};
    printf("L1-Norm of `vec` is %.1f\n", 
           l1_norm(vec, 3));
    return 0;
}
