#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the L2 norm of a given array of numbers.
 * 
 * @param x The parameter `x` is a pointer to an array of `double` values. It represents the vector for
 * which we want to calculate the L2 norm.
 * @param n The parameter `n` represents the size of the array `x`. It indicates the number of elements
 * in the array that need to be considered for calculating the L2 norm.
 * 
 * @return the square root of the sum of the squares of the elements in the array `x`.
 */
double l2_norm(double *x, int n) {
    int i;
    double res = 0.0;
    #pragma omp parallel for simd reduction (+ : res)
    for (i = 0; i < n; i++) {
        res += x[i] * x[i];
    }
    return sqrt(res);
}

/* Test function */
int main() {
    double vec[] = {0.1, -0.1, 0.5};
    printf("L2-Norm of `vec` is %f\n", 
           l2_norm(vec, 3));
    return 0;
}
