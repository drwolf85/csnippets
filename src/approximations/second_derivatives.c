#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Numerical approximation of the second derivative of $f(x)$ computed in $x = x_0$
 * 
 * @param x0 Value where to compute the second derivative of $f$
 * @param h Positive number for the length of the segment in $\mathbb{R}_+$
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}\to\mathbb{R}$
 * @return double 
 */
double deriv2(double x0, double h, double htype, double(*f)(double)) {
    double res = 0.0;
    double xl, xc, xr;
    xl = xc = xr = x0;
    xl += 2.0 * h * (1.0 - htype);
    xc += h * (1.0 - 2.0 * htype);
    xr -= 2.0 * h * htype;
    res = f(xl);
    res -= 2.0 * f(xc);
    res += f(xr);
    return res / (h * h);
}

/**
 * @brief Numerical approximation of the Hessian
 *        of $f: \mathbb{R}^n \to \mathbb{R}$ 
 *        computed in $x = x_0 \in \mathbb{R}^n$
 * 
 * @param hess Pointer to a vector where to store the Hessian
 * @param x0 Pointer to a vector of values where to compute the Hessian
 * @param n Positive integer for the length of $x_0$ and `grad`
 * @param h Positive number for the length of the segment along each coordinate
 * @param htype A number in [0, 1]. If 0.5, it will use the mid-point approximation
 * @param f A function $f:\mathbb{R}^n\to\mathbb{R}$
 */
void hessian(double *hess, double *x0, int n, double h, double htype, double(*f)(double *, int)) {
    int i, j, k;
    double *xnw, *xne, *xsw, *xse;
    double const ihh = 1.0 / (h * h);
    xnw = (double *) malloc(n * sizeof(double));
    xne = (double *) malloc(n * sizeof(double));
    xsw = (double *) malloc(n * sizeof(double));
    xse = (double *) malloc(n * sizeof(double));
    if (xne && xnw && xse && xsw && hess && x0) {
        memset(hess, 0, n * n * sizeof(double));
        memcpy(xne, x0, n * sizeof(double));
        memcpy(xnw, x0, n * sizeof(double));
        memcpy(xse, x0, n * sizeof(double));
        memcpy(xsw, x0, n * sizeof(double));

        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                xnw[i] += h * (1.0 - htype);
                xnw[j] += h * (1.0 - htype);
                xne[i] -= h * htype;
                xne[j] += h * (1.0 - htype);
                xsw[i] += h * (1.0 - htype);
                xsw[j] -= h * htype;
                xse[i] -= h * htype;
                xse[j] -= h * htype;
                k = i + n * j;
                hess[k] = f(xnw, n);
                hess[k] -= f(xne, n);
                hess[k] -= f(xse, n);
                hess[k] += f(xsw, n);
                hess[k] *= ihh;
                hess[j + n * i] = hess[k];
                xnw[i] = x0[i];
                xnw[j] = x0[j];
                xne[i] = x0[i];
                xne[j] = x0[j];
                xsw[i] = x0[i];
                xsw[j] = x0[j];
                xse[i] = x0[i];
                xse[j] = x0[j];
            }
        }
    }
}

/* Testing the functions above */
double myfun(double *x, int n) {
    int i;
    double res = 0.0;
    for (i = 0; i < n; i++) {
        res += x[i] * x[i];
    }
    return 1.0 / (1.0 + res);
}

int main() {
    int i;
    double x[] = {0.1, -0.5, -0.9};
    double hess[9] = {0};
    printf("Second derivative of exp(x) in x = 1: %f\n", deriv2(1.0, 1e-4, 0.5, exp));
    printf("Hessian of `myfun` in x = (");
    for (i = 0; i < 2; i++) {
        printf("%.1f, ", x[i]);
    }
    printf("%.1f):\n\t", x[i]);
    hessian(hess, x, 3, 1e-2, 0.5, myfun);
    for (i = 0; i < 9; i++) {
        printf("%.4f\t", hess[i]);
        if (i % 3 == 2) printf("\n\t");
    }
    return 0;
}
