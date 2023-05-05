/**
 * @file bernstein_smoother.c
 * @brief Implementation of the Bernstein polynomials and a
 *   smoother based on this one dimensional function approximation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * The function calculates the Bernstein polynomial of a given degree for a given function.
 * 
 * @param x x is a double precision floating point number that represents the input value for the
 * Bernstein polynomial. It should be between 0 and 1, inclusive.
 * @param f The parameter `f` is a pointer to a function that takes a double as input and returns a
 * double as output. This function is used to evaluate the function being approximated by the Bernstein
 * polynomial.
 * @param n n is the degree of the Bernstein polynomial. It determines the number of terms in the
 * polynomial and the accuracy of the approximation.
 * 
 * @return The function `bernstein_poly` returns a double value which is the result of evaluating the
 * Bernstein polynomial of degree `n` at the point `x` using the function `f` as the input function.
 */
double bernstein_poly(double x, double (*f)(double), int n) {
    int i;
    double omx, tmp, res = 0.0;
    n--;
    if (x < 0.0 || x > 1.0) { 
        res = nan(""); 
    }
    else if (x == 0.0 || x == 1.0) {
        res = (*f)(x);
    }
    else {
        omx = log(1.0 - x);
        x = log(x);
        #pragma omp parallel for simd private(tmp) reduction(+ : res)
        for (i = 0; i <= n; i++) {
            tmp = lgamma((double) (n + 1));
            tmp -= lgamma((double) (i + 1));
            tmp -= lgamma((double) (n - i + 1));
            tmp += x * (double) i;
            tmp += omx * (double) (n - i);
            res += exp(tmp) * (*f)((double) i / (double) n);
        }
    }
    return res;
}

/**
 * The function implements the Bernstein smoothing algorithm for a set of values using a given
 * parameter.
 * 
 * @param x The input value to be smoothed, which should be between 0 and 1 inclusive.
 * @param v v is a pointer to an array of doubles representing the values of a function at a set of
 * points. The function is being smoothed using the Bernstein polynomial method.
 * @param n The degree of the Bernstein polynomial, which needs to be equal to the number of control
 * points.
 * 
 * @return The function `bernstein_smooth` returns a double value which is the result of applying the
 * Bernstein polynomial smoothing function to the input value `x` using the coefficients in the array
 * `v` of length `n`.
 */
double bernstein_smooth(double x, double *v, int n) {
    int i;
    double omx, tmp, res = 0.0;
    n--;
    if (x < 0.0 || x > 1.0) { 
        res = nan(""); 
    }
    else if (x == 0.0) {
        res = v[0];
    }
    else if (x == 1.0) {
        res = v[n];
    }
    else {
        omx = log(1.0 - x);
        x = log(x);
        #pragma omp parallel for simd private(tmp) reduction(+ : res)
        for (i = 0; i <= n; i++) {
            tmp = lgamma((double) (n + 1));
            tmp -= lgamma((double) (i + 1));
            tmp -= lgamma((double) (n - i + 1));
            tmp += x * (double) i;
            tmp += omx * (double) (n - i);
            res += exp(tmp) * v[i];
        }
    }
    return res;
}

/* Test functions */
double myF(double x) {
    x *= M_PI;
    return sin(x)*cos(x);
}

int main() {
    double v[] = { 1.0, 0.5, 0.25, .125, 0.0};
    double sm, pl;
    int i;
    printf("m <- matrix(c(");
    for (i = 0; i < 100; i++) {
        sm = bernstein_smooth(0.01 * (double) i, v, 5);
        pl = bernstein_poly(0.01 * (double) i, myF, 10);
        printf("%f,%f,", sm, pl);
    }
    sm = bernstein_smooth(0.01 * (double) i, v, 5);
    pl = bernstein_poly(0.01 * (double) i, myF, 10);
    printf("%f,%f", sm, pl);
    printf("), ncol = 2, byrow = TRUE)\nplot(seq_len(101), m[,1L], type = \"l\")\n"
           "x11()\nplot(seq_len(101), m[,2L], type = \"l\")\n");
    return 0;
}
