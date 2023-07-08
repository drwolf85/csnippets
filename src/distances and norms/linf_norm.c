#include <stdio.h>
#include <math.h>
#include <omp.h>

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
