/**
 * @brief Averages according to Oscar Chisini's definition 
 *          f(x_1, ..., x_n) = f(x, ..., x)  
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS_TOLL 1e-10

/**
 * The above type defines a struct with a double value and a `size_t` index.
 * @property {double} v - The property "v" is a double precision floating point number that represents
 * the value of a vector element.
 * @property {size_t} i - `i` is a member variable of the `vec` struct and its data type is `size_t`.
 * It is used to store the index of a vector element.
 */
typedef struct vec {
    double v;
    size_t i;
} vec;

/**
 * The function compares two values in a struct and returns 0 if they are equal, 1 if the first value
 * is NaN, -1 if the second value is NaN, and 1 or -1 depending on which value is greater if neither
 * value is NaN.
 * 
 * @param a The parameter "a" is a void pointer that can be cast to a pointer of type "vec".
 * @param b The parameter "b" is a void pointer, which means it can point to any type of data. In this
 * specific function, it is being cast to a pointer of type "vec" using the line "vec *vb = (vec *)
 * b;". This means that "b" is expected
 * 
 * @return The function `cmp_val` returns an integer value that indicates the comparison result between
 * two `vec` structures.
 */
int cmp_val(const void *a, const void *b) {
    vec *va = (vec *) a;
    vec *vb = (vec *) b;
    if (fabs(va->v - vb->v) < EPS_TOLL) {
        return 0;
    }
    else if (isnan(va->v)) return 1;
    else if (isnan(vb->v)) return -1;
    else {
        return (int) (va->v > vb->v) * 2 - 1;
    }
}

/**
 * The function performs a binary search to find the root of a given function within a given interval.
 * 
 * @param y an array of data values
 * @param n The number of data values in the input array y.
 * @param fun The parameter `fun` is a pointer to a function that takes a double array `y` and a size_t
 * `n` as input and returns a double value. This function is used to compute a generic function for
 * each data value in the `y` array.
 * 
 * @return a double value, which is the root of a function found using the Chisini method.
 */
double chisini(double *y, size_t n, double (*fun)(double *, size_t)) {
    double *x, d, v, res = nan("");
    double w[3];
    const double ans = (*fun)(y, n);
    size_t i, j;
    vec *z;

    if (isnan(ans)) return res;

    x = (double *) malloc(n * sizeof(double));
    z = (vec *) malloc(n * sizeof(vec));
    if (x && z) {
        /* Compute generic function for each data value */
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                x[i] = y[j];
            }
           /* Relocation of the output */
           z[j].v = (*fun)(x, n) - ans;
           z[j].i = j;
        }
        /* Sorting the output values */
        qsort(z, n, sizeof(vec), cmp_val);
        if (z[0].v > 0.0) {
            /* If the minimum is greater then zero, 
               find initial configuration for a binary search */
            for (i = 0; i < n; i++) {
                x[i] = y[z[0].i] * 1.0001;
            }
            v = (*fun)(x, n) - ans;
            d = 1.0;
            do {
                *x -= d * (v - z[0].v) / (y[z[0].i] * 0.0001);
                for (i = 1; i < n; i++) {
                    x[i] = *x;
                }
                v = (*fun)(x, n) - ans;
                d *= 2.0;
            } while(v >= 0.0);
            w[0] = *x;
            w[2] = y[z[0].i];
        }
        else if (z[n-1].v < 0.0) {
            /* If the maximum is less then zero, 
               find initial configuration for a binary search */
            for (i = 0; i < n; i++) {
                x[i] = y[z[n-1].i] * 1.0001;
            }
            v = (*fun)(x, n) - ans;
            d = 1.0;
            do {
                *x += d * (v - z[n-1].v) / (y[z[n-1].i] * 0.0001);
                for (i = 1; i < n; i++) {
                    x[i] = *x;
                }
                v = (*fun)(x, n) - ans;
                d *= 2.0;
            } while(v <= 0.0);
            w[0] = y[z[n-1].i];
            w[2] = *x;
        }
        else { /* If the interval between minimum and maximum contains the zero */
            for (i = 0; z[i].v < 0.0; i++);
            w[0] = y[z[i - 1].i];
            w[2] = y[z[i].i];
        }
        /* Binary search */
        w[1] = (w[0] + w[2]) * 0.5;
        while (w[2] - w[0] > EPS_TOLL) {
            for (i = 0; i < n; i++) {
                x[i] = w[1];
            }
            v = (*fun)(x, n) - ans;
            if (v < 0.0) {
                w[0] = w[1];
            }
            else {
                w[2] = w[1];
            }
            w[1] = (w[0] + w[2]) * 0.5;
        }
        res = w[1];
    }
    free(x);
    free(z);
    return res;
}



/* Functions to test the validity of the approach above */
double test_fun(double *x, size_t n) {
    size_t i;
    double res = 1.0;
    for (i = 0; i < n; i++) {
        res = sqrt(res + exp(x[i]));
    }
    return res;
}

int main() {
    int i;
    double x[10] = {1.0, 0.2, -0.5, 0.6, -0.8, 
                   -0.7, -0.4, 2.5, -1.6, 0.8};
    double m = nan("");
    double res = test_fun(x, 10);
    printf("f(");
    for (i = 0; i < 9; i++) printf("%.1f, ", x[i]);
    printf("%.1f) = %f\n", x[i], res);
    m = chisini(x, 10, test_fun);
    for (i = 0; i < 10; i++) x[i] = m;
    res = test_fun(x, 10);
    printf("f(");
    for (i = 0; i < 9; i++) printf("%f, ", m);
    printf("%f) = %f\n", m, res);
    return 0;
}
