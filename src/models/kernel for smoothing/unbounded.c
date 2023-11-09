#include "kernels.h"

double laplacian_kernel(double x) {
    return 0.5 * exp(-fabs(x));
}

double silverman_kernel(double x) {
    double tmp = fabs(x) * M_SQRT1_2;
    return 0.5 * exp(-tmp) * sin(tmp + M_PI_4);
}

double gaussian_kernel(double x) {
    x = exp(-0.5 * x * x);
    return x / sqrt(2.0 * M_PI);
}
