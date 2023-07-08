#include <stdio.h>
#include <math.h>
#include <omp.h>

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
