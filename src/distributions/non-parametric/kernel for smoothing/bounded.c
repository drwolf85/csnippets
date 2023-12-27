#include "kernels.h"

double uniform_kernel(double x) {
    return 0.5 * ID_BOUND_1(x);
}

double triangular_kernel(double x) {
    return (1.0 - fabs(x)) * ID_BOUND_1(x);
}

double epanechnikov_kernel(double x) {
    return 0.75 * (1.0 - x * x) * ID_BOUND_1(x);
}

double quartic_kernel(double x) {
    double tmp = (1.0 - x * x);
    return (15.0 / 16.0) * tmp * tmp * ID_BOUND_1(x);
}

double triweight_kernel(double x) {
    double tmp = (1.0 - x * x);
    return (35.0 / 32.0) * tmp * tmp * tmp * ID_BOUND_1(x);
}

double tricube_kernel(double x) {
    double tmp = (1.0 - fabs(x * x * x));
    return (70.0 / 81.0) * tmp * tmp * tmp * ID_BOUND_1(x);
}

double cosine_kernel(double x) {
    return M_PI_4 * cos(M_PI_2 * x) * ID_BOUND_1(x);
}

double hyperbolic_cosine_kernel(double x) {
    double const l2sqr3 = log(2.0 + sqrt(3));
    double const factor = 1.0 / (4.0 - 2.0 * sinh(l2sqr3) / log(l2sqr3));
    double tmp = 2.0 - cosh(l2sqr3 * x);
    return factor * tmp * ID_BOUND_1(x);
}
