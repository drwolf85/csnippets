#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef DEBUG
void print_sq_mat(double *m, int n) {
	int i;
	for (i = 0; i < n * n; i++) {
		printf("%s%.08f%s", m[i] < 0 ? "" : " ", \
		       m[i], (i + 1) % n == 0 ? "\n" : " ");
	}
}

void print_sq_mat_e(double *m, int n) {
	int i;
	for (i = 0; i < n * n; i++) {
		printf("%s%.08e%s", m[i] < 0 ? "" : " ", \
		       m[i], (i + 1) % n == 0 ? "\n" : " ");
	}
}
#endif

/**
 * @brief Residuals of a linear model (based on OLS)
 * @param res empty vector to store the residuals (in output)
 * @param y response vector (example data for model output)
 * @param dta matrix of data (column-major format)
 * @param dim vector of dimension of `dta` matrix 
 */
static inline void lm_resid(double *res, double *y, double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && vec) {
        for (i = 0; i < dim[1]; i++) {
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] = dta[*dim * i + j];
            for (k = 0; k < i; k++) {
                tmp = 0.0;
                v = 0.0;
                for (j = 0; j < dim[0]; j++) {
                    tmp += q[*dim * k + j] * dta[*dim * i + j];
                    v += q[*dim * k + j] * q[*dim * k + j];
                }
                tmp /= v;
                #pragma omp for simd
                for (j = 0; j < dim[0]; j++) 
                    q[*dim * i + j] -= tmp * q[*dim * k + j];
            }
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            itmp = 1.0 / sqrt(tmp);
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *= itmp;
        }
        /* Computing least square residuals (Q^t y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * y[j];
            }
            vec[k] = tmp;
        }
        /* Computing least square residuals y - (QQ^t) y */
        for (j = 0; j < dim[0]; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = y[j] - tmp;
        }
    }
    free(q);
    free(vec);
}

/**
 * The function `qrR` computes the QR decomposition of a matrix `dta` of dimension `dim` and stores
 * the result in `r`
 * 
 * @param dta the data matrix
 * @param dim the dimensions of the matrix
 * @return a pointer to the output matrix
 */
static inline double * qrR(double *dta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q = NULL, *r = NULL;
    if (dim && dta) {
	    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
	    r = (double *) calloc(dim[1] * dim[1], sizeof(double));
	    if(q && r) for (i = 0; i < dim[1]; i++) {
	        #pragma omp for simd
		for (j = 0; j < dim[0]; j++)
		    q[*dim * i + j] = dta[*dim * i + j];
		for (k = 0; k < i; k++) {
		    tmp = 0.0;
		    v = 0.0;
		    for (j = 0; j < dim[0]; j++) {
		        tmp += q[*dim * k + j] * dta[*dim * i + j];
		        v += q[*dim * k + j] * q[*dim * k + j];
		    }
		    r[dim[1] * i + k] = tmp / sqrt(v);
		    tmp /= v;
		    #pragma omp for simd
		    for (j = 0; j < dim[0]; j++) 
		        q[*dim * i + j] -= tmp * q[*dim * k + j];
		}
		/* Normalization of the column vector */
		tmp = 0.0;
		for (j = 0; j < dim[0]; j++)
		    tmp += q[*dim * i + j] * q[*dim * i + j];
		tmp = sqrt(tmp);
		r[dim[1] * i + i] = tmp;
		itmp = 1.0 / tmp;
		#pragma omp for simd
		for (j = 0; j < dim[0]; j++)
		    q[*dim * i + j] *= itmp;
	    }
    }
    free(q);
    return r;
}

/**
 * The function `inverseUT` takes a square matrix `mat` and its dimension `n` as input, and returns
 * the inverse of the upper triangular matrix `mat` in place
 * 
 * @param mat the matrix to be inverted
 * @param nn the dimension of the matrix
 */
static inline void inverseUT(double *mat, int *nn) {
    int i, j, k, pos, n = *nn;
    double tmp;
    for (i = n; i > 0; i--) {
        pos = (n + 1) * (i - 1);
        mat[pos] = 1.0 / mat[pos];
        for (j = n - 1; j + 1 > i; j--) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * j + k] * mat[n * k + i - 1];
            }
            mat[n * j + i - 1] = tmp * (- mat[pos]);
        }
    }
}

/**
 * It takes a matrix and returns the outer product of the matrix with itself
 * 
 * @param mat a pointer to the first element of the matrix
 * @param nn the number of rows and columns of the matrix
 */
void outer_prod_UpperTri(double *mat, int *nn) {
    int i, j, k, n = *nn;
    double tmp;
    for (j = 0; j < n; j++) {
        for (i = 0; i <= j; i++) {
            tmp = 0.0;
            for (k = i; k < n; k++) {
                tmp += mat[n * k + j] * mat[n * k + i];
            }
            mat[n * j + i] = tmp;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            mat[n * i + j] = mat[n * j + i];
        }
    }
}

/**
 * @brief Variance of coefficient estimates of a linear model (based on OLS)
 * @param y response vector (example data for model output)
 * @param dta matrix of data (column-major format) - Input covariates
 * @param dim vector of dimension of `dta` matrix 
 * @return a pointer to the covariance matrix of the coefficient estimates
 */
extern double * lm_var_coef(double *y, double *dta, int *dim) {
	double *var = NULL, *res = NULL;
	double sigma2 = 0.0;
	int i, den;
	if (dim) den = dim[0] - dim[1];
	if (dim && dta && y && den > 0) {
		res = (double *) calloc(*dim, sizeof(double));
		var = qrR(dta, dim);
		#ifdef DEBUG
		printf("R:\n"); print_sq_mat(var, dim[1]);
		#endif
		if (var && res) {
			lm_resid(res, y, dta, dim);
			#ifdef DEBUG
			printf("Residuals: "); for (i = 0; i < *dim; i++) printf("%.09f ", res[i]); printf("\n");
			#endif
			inverseUT(var, &dim[1]);
			#ifdef DEBUG
			printf("R^-1:\n"); print_sq_mat(var, dim[1]);
			#endif
			outer_prod_UpperTri(var, &dim[1]);
			#ifdef DEBUG
			printf("R^-1%*%(R^-1)^t:\n"); print_sq_mat(var, dim[1]);
			#endif
			for (i = 0; i < *dim; i++) sigma2 += res[i] * res[i];
			sigma2 /= (double) den;
			#ifdef DEBUG
			printf("sigma sq = %f\n", sigma2);
			#endif
			#pragma omp for simd
			for (i = 0; i < dim[1] * dim[1]; i++) var[i] *= sigma2;
		}
		free(res);
	}
	return var;
}

#ifdef DEBUG
int main() {
	double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, -0.8869547, -0.105075, -1.236219, -0.2348659, 1.733807, -1.684883, -0.2730944, 0.1451159, -1.890664, 0.5631616};
	double y[] = {-1.053515, -1.86379, -3.528344, -0.1092058, -0.8494015};
	int dim[2] = {5, 3};
	double *var_betas = lm_var_coef(y, x, dim);
	printf("Var(\\hat{\\beta}): \n"); print_sq_mat_e(var_betas, 3);
	free(var_betas);
	return 0;
}
#endif

