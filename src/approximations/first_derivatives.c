#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Numerical approximation of the first derivative of $f(x)$ computed in $x = x_0$
 * 
 * @param x0 Value where to compute the first derivative of $f$
 * @param h Positive number for the length of the segment in $\mathbb{R}_+$
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}\to\mathbb{R}$
 * @return double 
 */
double deriv1(double x0, double h, double htype, double(*f)(double)) {
    double res = 0.0;
    double xl, xr;
    xl = xr = x0;
    xl += h * (1.0 - htype);
    xr -= h * htype;
    res = f(xl);
    res -= f(xr);
    return res / h;
}

/**
 * @brief Numerical approximation of the first directional derivative
 *        of $f: \mathbb{R}^n \to \mathbb{R}$ 
 *        computed in $x = x_0 \in \mathbb{R}^n$
 *        towards $\delta \in \mathbb{R}^n$
 * 
 * @param x0 Pointer to a vector of values where to compute $f'(x_0)$
 * @param n Positive integer for the length of $x_0$
 * @param delta Pointer to a vector of size `n` with the direction of the derivative
 * @param h Positive number for the length of the segment $\delta \in \mathbb{R}^n$
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}^n\to\mathbb{R}$
 * @return double 
 */
double directional_derivative(double *x0, int n, double *delta, double h, double htype, double(*f)(double *, int)) {
    double res = 0.0;
    int i;
    double *xl, *xr;
    xl = (double *) malloc(n * sizeof(double));
    xr = (double *) malloc(n * sizeof(double));
    if (xl && xr) {
        memcpy(xl, x0, n * sizeof(double));
        memcpy(xr, x0, n * sizeof(double));
        /* Normalize the direction */
        for (i = 0; i < n; i++) {
            res += delta[i] * delta[i];
        }
        res = sqrt(res);
        res = res > 0.0 ? 1.0 / res : 1.0;
        /* Compute points to approximate the derivative */
        for (i = 0; i < n; i++) {
            xl[i] += h * (1.0 - htype) * delta[i] * res;
            xr[i] -= h * htype * delta[i] * res;
        }
        res = f(xl, n);
        res -= f(xr, n);
    }
    free(xl);
    free(xr);
    return res / h;
}

/**
 * @brief Numerical approximation of the gradient
 *        of $f: \mathbb{R}^n \to \mathbb{R}$ 
 *        computed in $x = x_0 \in \mathbb{R}^n$
 *
 * @param grad Pointer to a vector where to store the gradient
 * @param x0 Pointer to a vector of values where to compute the gradient
 * @param n Positive integer for the length of $x_0$ and `grad`
 * @param h Positive number for the length of the segment along each coordinate
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}^n\to\mathbb{R}$
 */
 void gradient(double *grad, double *x0, int n, double h, double htype, double(*f)(double *, int)) {
    int i;
    double *xl, *xr;
    double const ih = 1.0 / h;
    xl = (double *) malloc(n * sizeof(double));
    xr = (double *) malloc(n * sizeof(double));
    if (xl && xr && grad) {
        memset(grad, 0, n * sizeof(double));
        memcpy(xl, x0, n * sizeof(double));
        memcpy(xr, x0, n * sizeof(double));
        for (i = 0; i < n; i++) {
            xl[i] += h * (1.0 - htype);
            xr[i] -= h * htype;
            grad[i] = f(xl, n);
            grad[i] -= f(xr, n);
            grad[i] *= ih;
            xl[i] = x0[i];
            xr[i] = x0[i];
        }
    }
    free(xl);
    free(xr);
 }

/* Testing the functions above */
double myfun(double *x, int n) {
    int i;
    double res = 0.0;
    for (i = 0; i < n; i++) {
        res += fabs(x[i]);
    }
    return 1.0 / (1.0 + res);
}

int main() {
    int i;
    double x[] = {0.1, 0.5, 0.9};
    double delta[] = {1.0, 1.0, 1.0};

    printf("First derivative of exp(x) in x = 1: %f\n", deriv1(1.0, 1e-6, 0.5, exp));

    printf("Directional derivative of `myfun` in x = (");
    for (i = 0; i < 2; i++) {
        printf("%.1f, ", x[i]);
    }
    printf("%.1f) ", x[i]);
    printf("along (");
    for (i = 0; i < 2; i++) {
        printf("%.0f, ", delta[i]);
    }
    printf("%.0f)", delta[i]);
    printf(": %f\n", directional_derivative(x, 3, delta, 1e-6, 0.5, myfun));

    printf("Gradient of `myfun` in x = (");
    for (i = 0; i < 2; i++) {
        printf("%.1f, ", x[i]);
    }
    printf("%.1f) ", x[i]);
    gradient(delta, x, 3, 1e-6, 0.5, myfun);
    printf(":\n\t(");
    for (i = 0; i < 2; i++) {
        printf("%f, ", delta[i]);
    }
    printf("%f)\n", delta[i]);
    return 0;
}
