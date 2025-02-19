#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * @brief Empirical Rectangular Prism Probability
 *
 * @param X Pointer to a matrix of data (stored in column-major format)
 * @param dimX Pointer to a vector with the dimensions of the matrix `X`
 * @param lb Pointer to a vector of lower bounds for the rectangular prims
 * @param ub Pointer to a vector of upper bounds for the rectangular prims
 * @return double
 */
double erpp(double *X, int *dimX, double *lb, double *ub) {
	unsigned i, cnt = 0;
	#pragma omp parallel for simd reduction(+ : cnt)
	for (i = 0; i < dimX[0]; i++) {
		char tmp = 1;
		for (unsigned j = 0; j < dimX[1] && tmp; j++) {
			tmp &= (char) (lb[j] < X[dimX[0] * j + i] && X[dimX[0] * j + i] <= ub[j]);
		}
		cnt += (unsigned) tmp;
	}
	return (double) cnt / (double) *dimX;
}

/**
 * @brief Empirical Closed Rectangular Prism Probability
 *
 * @param X Pointer to a matrix of data (stored in column-major format)
 * @param dimX Pointer to a vector with the dimensions of the matrix `X`
 * @param lb Pointer to a vector of lower bounds for the rectangular prims
 * @param ub Pointer to a vector of upper bounds for the rectangular prims
 * @return double
 */
double ecrpp(double *X, int *dimX, double *lb, double *ub) {
	unsigned i, cnt = 0;
	#pragma omp parallel for simd reduction(+ : cnt)
	for (i = 0; i < dimX[0]; i++) {
		char tmp = 1;
		for (unsigned j = 0; j < dimX[1] && tmp; j++) {
			tmp &= (char) (lb[j] <= X[dimX[0] * j + i] && X[dimX[0] * j + i] <= ub[j]);
		}
		cnt += (unsigned) tmp;
	}
	return (double) cnt / (double) *dimX;
}

/**
 * @brief Empirical Open Rectangular Prism Probability
 *
 * @param X Pointer to a matrix of data (stored in column-major format)
 * @param dimX Pointer to a vector with the dimensions of the matrix `X`
 * @param lb Pointer to a vector of lower bounds for the rectangular prims
 * @param ub Pointer to a vector of upper bounds for the rectangular prims
 * @return double
 */
double eorpp(double *X, int *dimX, double *lb, double *ub) {
	unsigned i, cnt = 0;
	#pragma omp parallel for simd reduction(+ : cnt)
	for (i = 0; i < dimX[0]; i++) {
		char tmp = 1;
		for (unsigned j = 0; j < dimX[1] && tmp; j++) {
			tmp &= (char) (lb[j] < X[dimX[0] * j + i] && X[dimX[0] * j + i] < ub[j]);
		}
		cnt += (unsigned) tmp;
	}
	return (double) cnt / (double) *dimX;
}

#ifdef DEBUG
int main() {
	double xmat[] = {0.2, 0.3, 0.1, 0.4, 0.7, \
			 0.4, 0.5, 0.9, 0.1, 0.8};
	int dimx[] = {5, 2};
	double lb [] = {0.2, 0.3};
	double ub [] = {0.4, 0.8};
	printf("Emp. RPP: %g\n", erpp(xmat, dimx, lb, ub));
	printf("Emp.Closed RPP: %g\n", ecrpp(xmat, dimx, lb, ub));
	printf("Emp.Open RPP: %g\n", eorpp(xmat, dimx, lb, ub));
	return 0;
}
#endif

