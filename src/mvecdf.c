#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define EPS_TOLL 1e-10
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

/**
 * It sorts the columns of a matrix, 
 * and then computes the cumulative distribution function
 * 
 * @param pr pointer to the output array (a vector of length `n`)
 * @param x a matrix of size `n` x `p`
 * @param n number of rows (or samples)
 * @param p number of columns in the matrix (or variables)
 * 
 * @return a pointer to an array of `size_t` with the order of 
 */
size_t * mvecdf(double *pr, double *x, size_t n, size_t p) {
    size_t i, j;
    double const invn = 1.0 / (double) n;
    values *v;
    size_t *o;

    v = (values *) malloc(n * p * sizeof(values));
    o = (size_t *) malloc(n * p * sizeof(size_t));
    if (v && o) {
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
        /* Set indexes in each column of an "order" matrix */
        #pragma omp parallel for private(i, j) collapse(2)
        for (j = 0; j < p; j++) {
            for (i = 0; i < n; i++) {
                o[n * j + v[n * j + i].i] = i;
            }
        }
        #pragma omp parallel for private(i, j)
        for (i = 0; i < n; i++) {
            pr[i] = 1.0;
            for (j = 0; j < p; j++) {
                pr[i] *= (0.5 + (double) o[n * j + i]) * invn;
            }
        }
        /* Prepaaring the data for a returning pointer of size `n` */
        v = (values *) realloc(v, n * sizeof(values));
        o = (size_t *) realloc(o, n * sizeof(size_t));
        /* Sorting the values of the vector `pr` and storing the indexes in `v` */
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            v[i].v = pr[i];
            v[i].i = i;
        }
        qsort(v, n, sizeof(values), cmp_values);
        /* Copy hte indices indexes in `o` */
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            o[i] = v[i].i;
        }
    }
    free(v);
    return o;
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
void mvecdf_prob(double *pr, double *x, int *_n, int *_p) {
    size_t i, j;
    size_t n = (size_t) *_n;
    size_t p = (size_t) *_p;
    double const invn = 1.0 / (double) n;
    values *v;
    size_t *o;

    v = (values *) malloc(n * p * sizeof(values));
    o = (size_t *) malloc(n * p * sizeof(size_t));
    if (v && o) {
        /* Organize the data of a matrix in a stracture values */
        // #pragma omp parallel for private(i, j) collapse(2)
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
        /* Set indexes in each column of an "order" matrix */
        #pragma omp parallel for private(i, j) collapse(2)
        for (j = 0; j < p; j++) {
            for (i = 0; i < n; i++) {
                o[n * j + v[n * j + i].i] = i;
            }
        }
        /* Compute the empirical cumulative distribution function */
        #pragma omp parallel for private(i, j)
        for (i = 0; i < n; i++) {
            pr[i] = 1.0;
            for (j = 0; j < p; j++) {
                pr[i] *= (0.5 + (double) o[n * j + i]) * invn;
            }
        }
    }
    free(v);
    free(o);
}

// dyn.load("test.so")
// pr <- double(n <- 500L)
// for (p in 1L + seq_len(7L)) {
//     x <- matrix(rnorm(n * p), n, p)
//     pr <- .C("mvecdf_prob", pr = pr, x, n, p, DUP = FALSE)$pr
//     cat("Number of variables:", p, "\n")
//     print(range(pr))
//     print(x[which.min(pr),])
//     print(x[which.max(pr),])
// }
// p <- 2L
// x <- matrix(rnorm(n * p), n, p)
// pr <- .C("mvecdf_prob", pr = pr, x, n, p, DUP = FALSE)$pr
// par(mfrow=c(2, 2))
// plot(x, cex = exp(pr - mean(pr) + 1) - 1)
// plot(x, type = "n")    
// text(x, labels = round(pr, 3))
// th <- apply(pnorm(x), 1, prod)
// plot(th, pr, xlim = 0:1, ylim = 0:1); abline(0:1, col=8)
// p <- 8L
// x <- matrix(rnorm(n * p), n, p)
// pr <- .C("mvecdf_prob", pr = pr, x, n, p, DUP = FALSE)$pr
// th <- apply(pnorm(x), 1, prod)
// plot(th, pr); abline(0:1, col=8)
