#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
/**
 * @brief Determinant of a 2x2 matrix
 *
 * @param A Pointer to a 2x2 matrix of data
 * @return double
 */
double det_2x2(double *A) {
    return A[0] * A[3] - A[1] * A[2];
}

/**
 * @brief Determinant of a 3x3 matrix
 *
 * @param A Pointer to a 3x3 matrix of data
 * @return double
 */
double det_3x3(double *A) {
    double res = A[0] * A[4] * A[8];
    res += A[3] * A[7] * A[2];
    res += A[6] * A[1] * A[5];
    res -= A[6] * A[4] * A[2];
    res -= A[0] * A[7] * A[5];
    res -= A[3] * A[1] * A[8];
    return res;
}

static inline void submatrix_det(double *sub, unsigned m, double *mat, unsigned n, unsigned j) {
    unsigned i, k;
    if (sub && mat && m == (n - 1)) {
    	for (k = 0; k < m; k++) {
            for (i = 0; i < j; i++) {
                sub[k * m + i] = mat[k * n + n + i];
            }
            for (i = j + 1; i < n; i++) {
                sub[k * m + i - 1] = mat[k * n + n + i];
            }
        }
    }
}

double det_nxn(double *A, unsigned n) {
    unsigned j;
    unsigned const m = n - 1;
    double *red;
    double res = 0.0;
    if (n == 3) {
        res = det_3x3(A);
    }
    else {
        red = (double *) malloc(m * m * sizeof(double));
        if (red) {
            for (j = 0; j < n; j++) {
                submatrix_det(red, m, A, n, j);
                res += (1.0 - (double) ((j & 1) << 1)) * A[j] * det_nxn(red, m);
            }
        }
        free(red);
    }
    return res;
}

/**
 * @brief LU decomposition
 *
 * @param A Pointer to a matrix stored in column-major format
 * @param n Number of columns (or rows) in matrix `A`
 *
 * @return The pointers to the Lower and Upper matrices
 */
static inline double ** LUdec(double *A, size_t n) {
    double **LU = NULL;
    bool both = true;
    size_t i, j, k;
    double tmp;
    double *a = (double *) malloc(n * n * sizeof(double));
    LU = (double **) calloc(2, sizeof(double *)); 
    if (__builtin_expect(LU && a, 1)) {
	for (i = 0; i < 2; i++) {
	    LU[i] = (double *) calloc(n * n, sizeof(double));
	    both = both && (bool) LU[i];
	}
	if (__builtin_expect(both, 1)) {
	    memcpy(a, A, n * n * sizeof(double));
	    /* Initialize the diagonal of the matrix L */
	    for (i = 0; i < n; i++) {
		LU[0][i * (n + 1)] = 1.0;
	    }
	    for (k = 0; k < n; k++) {
		tmp = a[k * (n + 1)]; /* Compute the pivots (diagonal of U) */
		LU[1][k * (n + 1)] = tmp;
		if (__builtin_expect(tmp == 0.0, 0)) break;
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
	   for (i = 0; i < 2; i++)
		if (__builtin_expect(LU[i] != NULL, 1))
		    free(LU[i]);
	   if (__builtin_expect(LU != NULL, 1)) free(LU);
	}
    }
    if (__builtin_expect(a != NULL, 1)) free(a);
    return LU;
}

extern double det(double *A, unsigned n) {
    unsigned i;
    double res = nan("");
    double **lu = LUdec(A, (size_t) n);
    if (__builtin_expect(lu != NULL, 1)) {
	if (__builtin_expect(lu[1] && n > 0, 1)) {
	    res = lu[1][0];
	    for (i = 1; i < n; i++) {
		res *= lu[1][i * (n + 1)];
	    }
	}
        if (__builtin_expect(lu[0] != NULL, 1)) free(lu[0]);
        if (__builtin_expect(lu[1] != NULL, 1)) free(lu[1]);
	free(lu);
    }
    return res;
}

extern double logdet(double *A, unsigned n) {
    unsigned i;
    double res = nan("");
    double **lu = LUdec(A, (size_t) n);
    if (__builtin_expect(lu != NULL, 1)) {
	if (__builtin_expect(lu[1] && n > 0, 1)) {
	    res = lu[1][0];
	    for (i = 1; i < n; i++) {
		res *= lu[1][i * (n + 1)];
	    }
	}
        if (__builtin_expect(lu[0] != NULL, 1)) free(lu[0]);
        if (__builtin_expect(lu[1] != NULL, 1)) free(lu[1]);
	free(lu);
    }
    return log(res);
}

#ifdef DEBUG
int main() {
	double my2x2[] = {1.0, 2.0, \
	                 -1.2, 1.1};
	double my3x3[] = {1.0, 2.0, 3.0, \
	                 -1.2, 1.1, 2.1, \
	                 -2.4, -4.2, 5.1};
    double my7x7[] = {0.1135069, 0.2518312, 0.9382962, 0.9759354, 0.9411557, 0.8613732, 0.8977938, 0.02803094, 0.2486456, 0.9911292, 0.9328471, 0.7964807, 0.698917, 0.9884764, 0.9003498, 0.7337028, 0.2864888, 0.07844389, 0.4853646, 0.3943964, 0.3309672, 0.3128202, 0.1932081, 0.9084621, 0.4118207, 0.9007977, 0.01523606, 0.006801641, 0.2852171, 0.9082577, 0.3066286, 0.8233232, 0.3885043, 0.8910223, 0.7980214, 0.7094773, 0.2900835, 0.7873994, 0.508668, 0.4854962, 0.5484769, 0.2022823, 0.5887926, 0.2410305, 0.7114156, 0.7971171, 0.5340627, 0.3526617, 0.4391141};
    double my10x10[] = {0.6633999, 0.2961706, 0.7282127, 0.3104062, 0.5260771, 0.9620794, 0.8337877, 0.2198294, 0.1016124, 0.1155931, 0.3671932, 0.1996878, 0.3340143, 0.3107977, 0.834269, 0.5742477, 0.08382831, 0.5530224, 0.6637939, 0.801384, 0.02587374, 0.9570121, 0.2837855, 0.4821114, 0.3384017, 0.005904601, 0.2273608, 0.05890343, 0.0806375, 0.2062787, 0.6633528, 0.6429189, 0.9196039, 0.7462055, 0.596692, 0.1232945, 0.1436624, 0.5201882, 0.5596214, 0.9529834, 0.1621655, 0.9883138, 0.6835381, 0.8361474, 0.2668868, 0.8427523, 0.1684795, 0.8792765, 0.6750625, 0.03964547, 0.0604508, 0.9965574, 0.1699632, 0.6014933, 0.6339295, 0.317738, 0.4976088, 0.08317769, 0.4044392, 0.6352655, 0.4667435, 0.6220974, 0.2025886, 0.06246104, 0.9312335, 0.8453667, 0.5874908, 0.1136461, 0.41208, 0.5161377, 0.3705263, 0.6614725, 0.2896652, 0.7339018, 0.4135488, 0.9346917, 0.1960334, 0.1048501, 0.144064, 0.5124006, 0.05869673, 0.4898512, 0.7581869, 0.8330355, 0.7223777, 0.5553477, 0.06718698, 0.8389868, 0.1104336, 0.02310638, 0.2241302, 0.06327994, 0.8527156, 0.0524084, 0.9229282, 0.5346431, 0.6196642, 0.002026024, 0.4377033, 0.9435216};
	printf("Det. test (2x2): %g\n", det_2x2(my2x2));
	printf("Det. test (3x3): %g\n", det_3x3(my3x3));
	printf("Det. test (7x7): %.9f\n", det_nxn(my7x7, 7UL));
	/* printf("Det. test (10x10): %.9f\n", det_nxn(my10x10, 10UL)); */
	printf("Det. test (10x10): %.9f\n", det(my10x10, 10UL));
	printf("Log-Det. test (10x10): %.9f\n", logdet(my10x10, 10UL));

	return 0;
}
#endif


