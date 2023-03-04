#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define EPS_TOLL 1e-10

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
int cmp_values(void *aa, void *bb) {
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

/**
 * It sorts the columns of a matrix, 
 * finds the minimum index among the columns of the matrix, 
 * and then computes the cumulative distribution function
 * 
 * @param pr pointer to the output array (a vector of length `n`)
 * @param x a matrix of size `n` x `p`
 * @param n number of rows (or samples)
 * @param p number of columns in the matrix (or variables)
 */
void mvecdf(double *pr, double *x, size_t n, size_t p) {
    size_t i, j;
    double const invn = 1.0 / (double) n;
    values *v;

    v = (values *) malloc(n * p * sizeof(values));
    if (v) {
        /* Organize the data of a matrix in a stracture values */
        #pragma omp parallel for private(i, j) collapse(2)
        for (j = 0; j < p; j++) {
            for (i = 0; i < n; i++) {
                v[n * j + i].v =  x[n * j + i];
                v[n * j + i].i = i;
            }
        }
        /* Sorting the data in each column of the matrix */
        #pragma omp parallel for private(j)
        for (j = 0; j < p; j++)
            qsort(&v[j * n], n, sizeof(values), cmp_values);
        /* Find the minimum index among the columns of the matrix */
        #pragma omp parallel for private(i, j)
        for (i = 0; i < n; i++) {
            for (j = 1; j < p; j++) {
                if (v[i].i > v[n * j + i].i) {
                    v[i].i = v[n * j + i].i;
                }
            }
        }
        /* Compute cumulative distribution function */
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            pr[i] = 0.5 + (double) v[i].i;
            pr[i] *= invn;
        }
    }
    free(v);
}
