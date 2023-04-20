#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define EPS_TOLL 1e-10

/**
 * The function calculates the Aitchison distance between two vectors of data that satisfy the simplex
 * properties.
 * 
 * @param x A pointer to an array of doubles representing the first probability distribution.
 * @param y The parameter `y` is a pointer to an array of `double` values representing the second point
 * in the Aitchison distance calculation.
 * @param n The parameter "n" represents the number of elements in the input arrays "x" and "y".
 * 
 * @return If the input data satisfies the simplex properties, the function returns the Aitchison
 * distance between the two input vectors. If the input data does not satisfy the simplex properties,
 * the function returns NaN (not a number).
 */
double aitchison_distance(double *x, double *y, size_t n) {
    double res = 0.0;
    double tmp = 0.0;
    size_t i = 0, j;
    #pragma omp parallel for simd reduction(+ : i, res, tmp)
    for (j = 0; j < n; j++) { /* Checking the format of the data in input */
        tmp += x[j];
        res += y[j];
        i += (x[j] < 0.0) + (y[j] < 0.0);
        i += (x[j] > 1.0) + (y[j] > 1.0);
    } /* If data satisfy the simplex poperties */
    if (i == 0 && fabs(tmp - 1.0) < EPS_TOLL && fabs(res - 1.0) < EPS_TOLL) { 
        res = 0.0;
        #pragma omp parallel for simd private(j, tmp) reduction(+ : res)
        for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
                tmp = log(x[i] / x[j]) - log(y[i] / y[j]);
                res += tmp * tmp;
            }
        }
        return sqrt(res);
    }
    else {
        return nan("");
    }
}

/* Test function */
int main() {
    double x[] = {0.1, 0.2, 0.5, 0.1, 0.1};
    double y[] = {0.2, 0.1, 0.4, 0.1, 0.2};
    size_t i;
    printf("Aitchison distance between x and y is %f\n", aitchison_distance(x, y, 5));
    return 0;
}
