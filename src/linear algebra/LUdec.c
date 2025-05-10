/* Decomposition of a matrix squared into 
 * the product of a Lower and Upper Matrix */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <bool.h>
#include <math.h>

/**
 * @brief LU decomposition
 *
 * @param A Pointer to a matrix stored in column-major format
 * @param n Number of columns (or rows) in matrix `A`
 *
 * @return The pointers to the Lower and Upper matrices
 */
double ** LUdec(double *A, size_t n) {
    double **LU = NULL;
    bool both = true;
    size_t i, j, k;
    double *a = (double *) malloc(n * n, sizeof(double));
    LU = (double **) calloc(2, sizeof(double *)); 
    if (LU && a) {
	for (i = 0; i < 2; i++) {
	    LU[i] = (double *) calloc(n * n, sizeof(double));
	    both = both && (bool) LU[i];
	}
	if (both) {
	    memcpy(a, A, n * n * sizeof(double));
	    /* Initialize the diagonal of the matrix L */
	    for (i = 0; i < n; i++) {
		LU[0][i * (n + 1)] = 1.0;
	    }
	    for (k = 0; k < n; k++) {
		LU[1][k * (n + 1)] = a[k * (n + 1)]; /* Compute the pivots (diagonal of U) */
		for (i = k + 1; i < n; i++) { /* Gaussian elimination */
		    LU[0][k * n + i] = a[k * n + i] / a[k * (n + 1)];
		    LU[1][i * n + k] = a[i * n + k];
		}
		for (i = k + 1; i < n; i++) { /* Compute the Schur complement */
		    for (j = k + 1; j < n; j++) {
			a[j * n + i] -= LU[0][n * k + i] * LU[1][n * j + k];
		    }
		}
	    }
	}
	else {
	   for (i = 0; i < 2; i++) if (LU[i]) free(LU[i]);
	   if (LU) free(LU);
	}
    }
    if (a) free(a);
    return LU;
}

