#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the Lp norm of a given array of numbers.
 * 
 * @param x An array of double values representing the vector for which we want to calculate the Lp
 * norm.
 * @param n The parameter `n` represents the length of the array `x`. It indicates the number of
 * elements in the array that need to be considered for the calculation of the Lp norm.
 * @param p The parameter "p" represents the exponent used in the calculation of the Lp norm. The Lp
 * norm is a way to measure the magnitude of a vector in a given space. It is defined as the p-th root
 * of the sum of the absolute values of the vector elements raised to the power
 * 
 * @return The function `lp_norm` returns the p-norm of the input array `x`.
 */
double lp_norm(double *x, int n, double p) {
    int i;
    double res = 0.0;
    #pragma omp parallel for simd reduction (+ : res)
    for (i = 0; i < n; i++) {
        res += pow(fabs(x[i]), p);
    }
    return pow(res, 1.0 / p);
}

/* Test function */
int main() {
    double vec[] = {0.1, -0.1, 0.5};
    double const p = 0.5;
    printf("Lp-Norm of `vec` with p = %.1f is %.1f\n", 
           p, lp_norm(vec, 3, p));
    return 0;
}
