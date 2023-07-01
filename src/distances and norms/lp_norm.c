#include <stdio.h>
#include <math.h>
#include <omp.h>

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
