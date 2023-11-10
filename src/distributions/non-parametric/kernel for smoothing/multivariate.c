#include "kernels.h"

double spherical_kernel(double *x, int p) {
    int i;
    double tmp = 0.0;
    for (i = 0; i < p; i++) tmp += x[i] * x[i];
    tmp = sqrt(fabs(tmp));
    return gamma(2.0 + 0.5 * p) * pow(M_SQRT2, (double) p) * (1.0 - tmp * tmp) * ID_BOUND_1(tmp);
}

double product_1_kernel(double *x, int p, double (*K)(double)) {
    int i;
    double tmp = 1.0;
    for (i = 0; i < p; i++) tmp *= K(x[i]);
    return tmp;
}


double product_p_kernels(double *x, int p, double (**K)(double)) {
    int i;
    double tmp = 1.0;
    for (i = 0; i < p; i++) tmp *= K[i](x[i]);
    return tmp;
}

double additive_1_bounded_kernel(double *x, int p, double (*K)(double)) {
    int i, j;
    double tmp, res = 0.0;
    for (i = 0; i < p; i++) {
        tmp = K(x[i]);
        for (j = 0; j < i; j++) tmp *= ID_BOUND_1_STRICT(x[j]);
        for (j++; j < p; j++) tmp *= ID_BOUND_1_STRICT(x[j]);
        res += tmp;
    }
    return res / ((double) p * pow(2.0, (double) (p - 1)));
}

double product_p_bounded_kernels(double *x, int p, double (**K)(double)) {
    int i, j;
    double tmp, res = 0.0;
    for (i = 0; i < p; i++) {
        tmp = K[i](x[i]);
        for (j = 0; j < i; j++) tmp *= ID_BOUND_1_STRICT(x[j]);
        for (j++; j < p; j++) tmp *= ID_BOUND_1_STRICT(x[j]);
        res += tmp;
    }
    return res / ((double) p * pow(2.0, (double) (p - 1)));
}
