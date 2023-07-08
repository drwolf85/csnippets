#include <stdio.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the L-infinity norm of an array of doubles.
 * 
 * @param x x is a pointer to an array of double values. It represents the input vector for which we
 * want to calculate the L-infinity norm.
 * @param n The parameter `n` represents the size of the array `x`. It indicates the number of elements
 * in the array that need to be considered for calculating the L-infinity norm.
 * 
 * @return The function `linf_norm` returns the maximum absolute value of the elements in the array
 * `x`.
 */
double linf_norm(double *x, int n) {
    int i;
    double res = fabs(*x);
    double tmp;
    #pragma omp parallel for private(tmp) reduction (max : res)
    for (i = 1; i < n; i++) {
        tmp = fabs(x[i]);
        if (tmp > res) res = tmp;
    }
    return res;
}

/* Test function */
int main() {
    double vec[] = {0.1, -0.1, 0.5};
    printf("L(inf)-Norm of `vec` is %.1f\n", 
           linf_norm(vec, 3));
    return 0;
}
