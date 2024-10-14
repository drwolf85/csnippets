#include <stdio.h>

/**
 * @brief B-spline basis function (using recursive definition)
 * @param x A double-precition floating-point number where to evaluate the basis function
 * @param i Integer number for the position of the knot in the array associated to the pointer `*knots_vec`
 * @param k Integer number for the order of the spline (e.g., `k = 4` is used for a cubic spline)
 * @param knots_vec Pointer to a preallocated array of knots of length `n + K`, i.e., by assuming `n` basis (for `i = 0, ..., n - 1`) and a B-splin order `K` 
 * @return A double-precition floating-point number with the value of the basis function
 */
double Bspline_basis(double x, unsigned i, unsigned k, double *knots_vec) {
	double lnum, rnum;
	if (k <= 1) return (double) (x >= knots_vec[i] && x < knots_vec[i + 1]);
	lnum = x - knots_vec[i];
	lnum /= knots_vec[i + k - 1] - knots_vec[i];
	lnum *= Bspline_basis(x, i, k - 1, knots_vec);
	rnum = knots_vec[i + k] - x;
	rnum /= knots_vec[i + k] - knots_vec[i + 1];
	rnum *= Bspline_basis(x, i + 1, k - 1, knots_vec);
	return lnum + rnum;
}

#ifdef DEBUG
#define N 10
#define K 5
#define NpK (N + K)
#define STEP 0.1

#define MY_EPS 1e-17

#include <stdlib.h>
#include <time.h>

int cmp_dbl(const void *aa, const void *bb) {
	double a = *(double *) aa;
	double b = *(double *) bb;
	return (a > b) * 2 - 1;
}

int main() {
	int i;
	double x, tot, tmp;
	double myknots[NpK] = {0.0};
	srand(time(NULL));
	
	/* Initialization of the knots */
	for (i = 0; i < NpK; i++) {
		myknots[i] = (double) rand() * (1.0 / (double) RAND_MAX);
		myknots[i] *= 1.0 + 2.0 * STEP;
		myknots[i] -= STEP;
	}
	qsort(myknots, NpK, sizeof(double), cmp_dbl);
	
	/* Testing */
	printf("Testing B-Spline Basis Function:\n");
	for (x = STEP; x < 1.0 - STEP - MY_EPS; x += STEP) {
		tot = 0.0;
		for (i = 0; i < N; i++) {
			tot += tmp = Bspline_basis(x, i, K, myknots);
			printf("B_{%d, %d}(%.1f) = %f \n", i, K, x, tmp);
		}
		printf("Total = %f\n\n", tot); /* This assumes B-spline weights equal to one */
	}
	return 0;
}
#endif

