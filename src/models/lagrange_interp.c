/**
 * @file lagrange_interp.c
 * @brief Lagrange interpolation of one-dimentional functions  * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double lagrange_interp(double x, double *xv, double *yv, size_t n) {
    double res = nan("");
    double over, under;
    size_t i, j = 0;

    /* Checking if all values in `xv` belongs to the interval [-1, 1] */
    #pragma omp parallel for simd reduction(+ : j)
    for (i = 0; i < n; i++) {
        j += xv[i] < -1.0;
        j += xv[i] > 1.0;
    }
    j += x > 1.0;
    j += x < -1.0;
    if (j == 0) { /* Interpolate */
        res = 0.0;
        #pragma omp parallel for simd private(j, over, under) reduction(+ : res)
        for (i = 0; i < n; i++) {
            over = 1.0;
            under = 1.0;
            for (j = 0; j < n; j++) {
                over *= (double) (j == i) + (double) (j != i) * (x - xv[j]);
                under *= (double) (j == i) + (double) (j != i) * (xv[i] - xv[j]);
            }
            res += yv[i] * over / under;
        }
    }
    return res;
}

/* Testing funciton */
#define N 101

int main() {
    double x[] = {-1.0, -0.9, -.42, .26, 0.4, 0.7, 1.0, 2.0};
    double y[] = {-10.0, -5.0, -.4, 1.2, 2.2, 5.2, 9.0, 0.0};
    double xx[] = {-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0};
    double yy[] = {-10.0, -5.0, 1.0, 2.0};
    double v;
    size_t i;

    printf("TEST WITH AT LEAST ONE DATA POINT OUTSIDE THE INTERVAL [-1, 1]"
           "Interpolated value in %.2f is %f\n\n"
           "TEST WITH ALL DATA IN [-1,1]\n", -.5, lagrange_interp(-.5, x, y, 8));
    for (i = 0; i < N; i++) {
        v = (double) i * 2.0 / (double) (N - 1) - 1.0;
        printf("Interpolated value in %f is %f \n", v, lagrange_interp(v, x, y, 7));
    }
    printf("\n\n");

    printf("# R command\n dta <- matrix(c(");
    v = -1.0;
    printf("%f, %f", v, lagrange_interp(v, xx, yy, 4));
        for (i = 1; i < N; i++) {
        v = (double) i * 2.0 / (double) (N - 1) - 1.0;
        printf(", %f, %f", v, lagrange_interp(v, xx, yy, 4));
    }
    printf("), ncol = 2, byrow = TRUE)\n");
    printf("plot(dta, type = 'l')\n");
    return 0;
}
