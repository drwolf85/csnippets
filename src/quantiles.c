#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define EPS_TOLL 1e-10

double minimum_value(double *x, size_t n) {
    size_t j, i = 0;
    double res = *x;
    while (isnan(res) && i < n) {
        i++;
        res = x[i];
    }
    j = i;
    for (i = j; i < n; i++) {
        if (!isnan(x[i])) {
            res += (res > x[i]) * (x[i] - res);
        }
    }
    return res;
}

double maximum_value(double *x, size_t n) {
    size_t i = 0;
    double res = *x;
    while (isnan(res) && i < n) {
        i++;
        res = x[i];
    }
    for (i = 0; i < n; i++) {
        if (!isnan(x[i])) {
            res += (res < x[i]) * (x[i] - res);
        }
    }
    return res;
}

double range(double *x, size_t n) {
    double minv = *x, maxv = *x;
    double res = nan("");
    size_t j, i = 0;
    /* Finding sample min and max */
    while (isnan(minv) && i < n) {
        i++;
        minv = x[i];
        maxv = x[i];
    }
    j = i;
    for (i = j; i < n; i++) {
        if (!isnan(x[i])) {
            minv += (minv > x[i]) * (x[i] - minv);
            maxv += (maxv < x[i]) * (x[i] - maxv);
        }
    }
    /* Compute the range of the sample */
    if (!isnan(minv) && !isnan(maxv)) res = maxv - minv;
    return res;
}

/**
 * A values is a struct with a `double` and a `size_t`.
 * @property {double} v - the value of the element
 * @property {size_t} i - the index of the value in the original array
 */
typedef struct {
    double v;
    size_t i;
} values;

/**
 * It compares two values, and returns -1 if the first 
 * is less than the second, 0 if they are equal,
 * and 1 if the first is greater than the second
 * 
 * @param aa the first value to compare
 * @param bb the value to compare to
 * 
 * @return the difference between the two values
 */
int cmp_values(const void *aa, const void *bb) {
    values *a = (values *)aa;
    values *b = (values *)bb;
    if (fabs(a->v - b->v) < EPS_TOLL) {
        return 0;
    }
    else if (isnan(a->v)) return 1;
    else if (isnan(b->v)) return -1;
    else {
        return (int) (a->v > b->v) * 2 - 1;
    }
}

double quantile(double *x, size_t n, double prob) {
    double target;
    double res = nan("");
    size_t i, nnan = 0;
    long pos;
    values *v;

    v = (values *) malloc(n * sizeof(values));
    if (v) {
        #pragma omp parallel for simd reduction(+ : nnan)
        for (i = 0; i < n; i++) {
            v[i].v = x[i];
            v[i].i = i;
            nnan += isnan(x[i]);
        }
        nnan = n - nnan;
        qsort(v, n, sizeof(values), cmp_values);
        /* Using linear interpolation between points */
        target = prob * (double) nnan;
        pos = (long) target - 1;
        pos *= (pos > 0);
        res = v[pos].v;
        prob = target - (double) ((size_t) target);
        res += prob *(v[pos + 1].v - res);
    }
    free(v); 
    return res;
}

/* Testing function */
int main() {
    double x[50] = { 1.04612564, -1.00765414, -0.95323787,  0.89483371, -0.74488022,  0.06825251,
                    -2.50374633, -0.70162665,  0.10846912,  0.86080390,  0.27829369, -1.29050847,
                     0.72689336, -1.22362417, -0.63146275, -3.02374906, -1.13205022,  0.25674405,
                    -0.34068293, -0.73221643,  1.45566137, -0.31540502,  0.14347480, -0.74859040,
                     0.21147942, -1.20158366, -0.60672815, -1.43170568,  0.67910960, -0.24818458,
                     1.02881213, -0.74976112,  0.99874893, -0.35633793, -0.58590449, -0.56258784,
                     0.47390399, -0.27336268, -0.19206190, -0.53284985, -0.87518074,  2.08260175,
                     0.52642028,  0.25721542,  0.59677736,  0.12434588, -0.02864796, -0.75645200,
                     0.02485415,  1.09674453 };
    printf("Minimum of x = %f\n", minimum_value(x, 50));
    printf("Maximum of x = %f\n", maximum_value(x, 50));
    printf("Range of x = %f\n", range(x, 50));
    printf("Third quartile of x = %f\n", quantile(x, 50, 0.75));
    return 0;
}
