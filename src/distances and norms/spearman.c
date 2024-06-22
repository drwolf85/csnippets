#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN_EPS 1e-40

typedef struct _m_vec_w_idx {
    double v;
    size_t i;
} vwi;


/**
 * The function `cmp_vwi` compares two structures of type `vwi` based on their `i` and `v` members.
 * 
 * @param aa The parameter `aa` is a pointer to a constant void type, which is then cast to a pointer
 * of type `vwi`.
 * @param bb The `bb` parameter in the `cmp_vwi` function is a pointer to a constant void type. It is
 * being cast to a pointer of type `vwi` in the function to compare two structures of type `vwi`.
 * 
 * @return The `cmp_vwi` function is a comparison function typically used with `qsort` or other sorting
 * algorithms. It compares two elements of type `vwi` based on their `i` and `v` fields. The function
 * returns an integer value that indicates the relative order of the two elements.
 */
int cmp_vwi(const void *aa, const void *bb) {
    int res = 0;
    vwi *a = (vwi *) aa;
    vwi *b = (vwi *) bb;
    res = (a->i > b->i) * 2 - 1;
    res *= (a->v == b->v);
    res = (a->v > b->v && res == 0) * 2 - 1;
    return res;
}

/**
 * The function calculates the Spearman's distance between two vectors of double values.
 * 
 * @param x A pointer to an array of double values representing the first vector.
 * @param y The parameter `y` is a pointer to a double array containing the second vector for which the
 * Spearman's distance is being calculated.
 * @param n The parameter "n" represents the size of the arrays "x" and "y", which are the input arrays
 * containing the vectors for which the Spearman's distance is being calculated.
 * 
 * @return the Spearman's distance (based on Pearson's correlation) between two vectors represented by arrays of double values.
 */
double spearman_distance(double *x, double *y, size_t n) {
    size_t i;
    vwi *rx, *ry;
    double rs = nan(""), tmp;
    rx = (vwi *) malloc(n * sizeof(vwi));
    ry = (vwi *) malloc(n * sizeof(vwi));
    if (rx && ry) {
        rs = 0.0;
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            rx[i].v = x[i];
            rx[i].i = i;
            ry[i].v = y[i];
            ry[i].i = i;
        }
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                qsort(rx, n, sizeof(vwi), cmp_vwi);
            }
            #pragma omp section
            {
                qsort(ry, n, sizeof(vwi), cmp_vwi);
            }
        }
        #pragma omp barrier
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            rx[rx[i].i].v = (double) i;
            ry[ry[i].i].v = (double) i;
        }
        #pragma omp parallel for private(i, tmp) reduction(+ : rs)
        for (i = 0; i < n; i++) {
            tmp = rx[i].v - ry[i].v;
            rs += tmp * tmp;
        }
        rs *= 6.0;
        rs /= (double) (n * (n * n - 1));
    }
    free(rx);
    free(ry);
    return 1.0 - rs;
}

/* Test function */
int main() {
    double x[] = {1.0, 2.0, 0.5, -0.5, -2.0};
    double y[] = {-1.1, 1.9, 0.4, -0.4, 1.0};
    double cd = spearman_distance(x, y, 5);
    printf("Computed Spearman's distance is %f\n", cd);
    return 0;
}

