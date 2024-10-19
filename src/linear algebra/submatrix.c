#include <stdio.h>
#include <stdlib.h>

double * submatrix(double *mat, unsigned n, unsigned row_id, unsigned col_id) {
    double *sub; /* Assumes col-major */
    unsigned const m = n - 1;
    unsigned i, k;
    sub = (double *) malloc(m * m * sizeof(double));
    if (sub && mat && m == (n - 1)) {
#define COPY_ROWS(val) { \
           for (i = 0; i < row_id; i++) { \
                sub[(k - (val)) * m + i] = mat[k * n + i]; \
            } \
            for (i = row_id + 1; i < n; i++) { \
                sub[(k - (val)) * m + i - 1] = mat[k * n + i]; \
            }}
    	for (k = 0; k < col_id; k++) {
    		COPY_ROWS(0);
        }
    	for (k = col_id + 1; k < n; k++) {
    		COPY_ROWS(1);
        }
    }
    return sub;
}

#ifdef DEBUG
static inline void print_sq_mat(double *A, unsigned n) {
	unsigned i, j;
	if (A) for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			printf("%f ", A[i + n * j]);
		printf("\n");
	}
}

int main() {
	double mat[] = {0.1741913, 0.8875169, 0.22515, 0.3964817, 0.5812981, 0.07996037, 0.9616732, 0.2191872, 0.2933022, 0.1438964, 0.07301235, 0.4013058, 0.5816609, 0.9829047, 0.8953716, 0.987558, 0.05223757, 0.5067601, 0.3287184, 0.6537518, 0.9504469, 0.3951467, 0.4181644, 0.27438, 0.04493142};
	double *sub;
	printf("Original matrix:\n");
	print_sq_mat(mat, 5UL);
	sub = submatrix(mat, 5L, 1UL, 2L);
	printf("\nSubmatrix by removing Row 2 and Column 3:\n");
	if (sub) print_sq_mat(sub, 4UL);
	free(sub);
	return 0;
}
#endif

