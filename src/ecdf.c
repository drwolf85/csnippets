#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Sort vector of indices
 * 
 * @param idx vector of indices
 * @param vec vector of positive integer data at 64 bits
 * @param n length of the two vectors described above
 */
void sort_index(size_t *idx, uint64_t *vec, size_t n) {
    size_t i, *tmp;
    char k;
    uint8_t c;
    size_t h[256], ch[256];
    tmp = (size_t *) malloc(n * sizeof(size_t));
    if (tmp) {
        for (k = 0; k < 8; k++) {
            for (i = 0; i < 256; i++) h[i] = 0; /* Reset the histogram */
            for (i = 0; i < n; i++) { /* Histogram */
                tmp[i] = idx[i];
                c = (uint8_t) ((vec[tmp[i]] >> (8 * k)) & 0xFF);
                h[c]++;
            }
            ch[0] = 0;
            for (i = 1; i < 256; i++) /* Compute starting positions */
                ch[i] = ch[i-1] + h[i-1];
            for (i = 0; i < n; i++) { /* Sorting (based on histogram) */
                c = (uint8_t) ((vec[tmp[i]] >> (8 * k)) & 0xFF);
                idx[ch[c]] = tmp[i];
                ch[c]++;
            }
        }
    }
    free(tmp);
}

/**
 * @brief Empirical cumulative distribution function (for integers)
 * 
 * @param y empty vector where to store the results
 * @param x vector of integer data
 * @param n length of the two vectors described above
 */
void ecdf_int(double *y, int *x, size_t n) {
    size_t i, c, *idx;
    int64_t *v;
    int m = *x;
    double const inp1 = 1.0 / (double) n;
    
    idx = (size_t *) malloc(n * sizeof(size_t));
    v = (int64_t *) malloc(n * sizeof(int64_t));

    if (idx && v) {
        for (i = 0; i < n; i++) {
            idx[i] = i; /* Initialize the indices */
            v[i] = (int64_t) x[i]; /* Copy data into temporary vector */
            m += (x[i] - m) * (int) (x[i] < m); /* Compute the minimum value */
        }
        for (i = 0; i < n; i++) v[i] -= (int64_t) m; /* Adjust the data before sorting */
        sort_index(idx, (uint64_t *) v, n); /* sort the vector of indices*/
        for (i = 0; i < n; i++) { /* Compute the ECDF */
            for (c = 0; i + c < n; c++) /* Check for duplicate values */
                if (v[idx[i]] != v[idx[i+c]]) break;
            for (c += i; i < c; i++) {
                    y[idx[i]] = inp1 * (double) c;
            }
            i--;
        }
    }
    free(idx);
    free(v);
}

/**
 * @brief Empirical cumulative distribution function (for real numbers)
 * 
 * @param y empty vector where to store the results
 * @param x vector of real numbers (data)
 * @param n length of the two vectors described above
 */
void ecdf_double(double *y, double *x, size_t n) {
    size_t i, c, *idx;
    double z;
    uint64_t *v;
    double const inp1 = 1.0 / (double) n;
    
    idx = (size_t *) malloc(n * sizeof(size_t));
    v = (int64_t *) malloc(n * sizeof(int64_t));

    if (idx && v) {
        for (i = 0; i < n; i++) {
            idx[i] = i; /* Initialize the indices */
            z = -(x[i] >= 0.0 ? x[i] : 1.0 / x[i]); /* Fix the order for IEEE format */
            v[i] = *((uint64_t *) &z); /* Copy data into temporary vector */
        }
        sort_index(idx, v, n); /* sort the vector of indices*/
        for (i = 0; i < n; i++) { /* Compute the ECDF */
            for (c = 0; i + c < n; c++) /* Check for duplicate values */
                if (v[idx[i]] != v[idx[i+c]]) break;
            for (c += i; i < c; i++) {
                    y[idx[i]] = inp1 * (double) c;
            }
            i--;
        }
    }
    free(idx);
    free(v);
}

// #define N 20
// int main() {
//     int i, x[] = {-2,0,0,1,910,1,2,-5,1,11, -25,30,50,13,9,10,22,-15,91,41};
//     double ecdf[N] = {0.0};
//     double y[] = {4.7,-5.1,0.0,1.0,-2.3,3.2,-6.7,7.8,-8.9,-9.2, 0.0,1.0,-2.3,3.2,-4.7,-5.1,6.7,-7.8,8.9,9.2};
//     ecdf_int(ecdf, x, N);
//     printf("Integers:\n");
//     for (i = 0; i < N; i++)
//         printf("%02d: %0.2f (for %d)\n", i, ecdf[i], x[i]);
//     ecdf_double(ecdf, y, N);
//     printf("\nReals:\n");
//     for (i = 0; i < N; i++)
//         printf("%02d: %0.2f (for %.2f)\n", i, ecdf[i], y[i]);
//     return 0;
// }
