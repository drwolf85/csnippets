#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double wt_second_moment(double *x, double *w, size_t n) {
    size_t i;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for simd reduction(+ : nnan, res)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i]) && !isnan(w[i])) {
            nnan += w[i];
            res += x[i] * x[i] * w[i];
        }
    }
    res /= nnan;
    return res;
}

double wt_var(double *x, double *w, size_t n) {
    size_t i;
    double sum = 0.0;
    double res = 0.0;
    double nnan = 0.0;
    #pragma omp parallel for simd reduction(+ : nnan, res, sum)
    for (i = 0; i < n; i++) {
        if (!isnan(x[i]) && !isnan(w[i])) {
            nnan += w[i];
            sum += x[i] * w[i];
            res += x[i] * x[i] * w[i];
        }
    }
    sum /= nnan;
    return res / nnan - sum * sum;
}

#ifdef DEBUG
/* Testing function */
int main () {
    double x[5] = { 0.0, 0.0, 1.0, 2.1, 4.5};
    double y[50]= { 0.0, 0.0, 1.0, 2.1, 4.5, 0.0, 1.0, 2.1, -4.5, 8.1,
                    0.0, 0.0, -1.0, 2.1, 4.5, 0.0, -1.0, 2.1, 4.5, 8.1,
                    0.0, 0.0, 1.0, -2.1, 4.5, 0.1, -1.0, -2.1, 4.5, -8.1,
                    0.0, 0.0, -1.0, 2.1, -4.5, 0.2, 1.0, 2.1, 4.5, 8.1,
                    0.0, 0.0, 1.0, -2.1, -4.5, 0.3, 1.0, -2.1, -4.5, 8.1};
    double w[50] = {1.041145, 0.3860256, 0.5740764, 0.7489271, 0.8450305, 0.6326457, 0.06026413, 2.46475, 0.1819105, 0.2178724, 1.691086, 1.097456, 0.970776, 0.6907987, 0.619246, 1.128712, 0.233283, 2.339812, 1.869066, 0.5534546, 1.800087, 0.2466242, 0.08780705, 0.6826522, 0.3012737, 0.1385971, 0.639267, 0.661176, 0.1787161, 1.365459, 1.098764, 2.163901, 0.7257147, 0.6886508, 0.4545968, 0.1530065, 0.1483386, 0.2171386, 0.2586661, 0.2430547, 0.9127081, 1.045831, 1.196502, 0.08143756, 0.1629122, 0.5977927, 0.7397318, 0.1735944, 1.042269, 1.002078};
    printf("Weighted second moment of x = %f\n", wt_second_moment(x, w, 5));
    printf("Weighted variance of x = %f\n", wt_var(x, w, 5));
    printf("Weighted variance of y = %f\n", wt_var(y, w, 50));
    return 0;
}
#endif

