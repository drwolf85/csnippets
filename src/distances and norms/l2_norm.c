#include <stdio.h>
#include <math.h>
#include <omp.h>

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
