#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5
#define STEP_DERIV 0.0125
#define MAX_LEARNING_RATE 1.99
#define MIN_LEARNING_RATE 1e-9

typedef struct vec {
	double v;
	int i;
} vec;

/** Progressive learning rate
 *
 * @param i Number of current iteration (between 0 and `max_iter`)
 * @param max_iter Number of maximum iterations
 *
 * @return double
 */
double get_learning_rate(unsigned i, unsigned max_iter) {
	double res = log(MIN_LEARNING_RATE / MAX_LEARNING_RATE);
	max_iter--;
	res *= (double) i / (double) max_iter;
	return exp(res) * MAX_LEARNING_RATE;
}

/** Comparison function between two `vec` items
 *
 * @param aa Pointer to the first `vec` structure to compare
 * @param aa Pointer to the second `vec` structure to compare
 * 
 * @return int
 */
static int cmp_vec(void const *aa, void const *bb) {
	vec a = *(vec *) aa;
	vec b = *(vec *) bb;
	return 2 * (int) (a.v < b.v) - 1; /* Reverse the order */
}

/** Function to update the gradient
 *
 * @param grd Pointer to a vector of `vec` structures 
 * @param param Pointer to a vector of parameters
 * @param len Pointer to the length of the `param` vector
 * @param info a pointer to a structu3re that contains the data and other information
 * @param objf Pointer to the objective function
 */
static void update_grad(vec *grd, double *param, int *len, void *info,
                        double (*objf)(double *, int *, void *)) {
	int i;
	double newf;
	double orgx;
	if (grd && param && len && objf) {
		for (i = 0; i < *len; i++) {
			grd[i].i = i;
			orgx = param[i];
			param[i] = orgx + STEP_DERIV;
			newf = objf(param, len, info);
			param[i] = orgx - STEP_DERIV;
			newf -= objf(param, len, info);
			param[i] = orgx;
			grd[i].v = fabs(newf / (2.0 * STEP_DERIV));
		}
		qsort(grd, *len, sizeof(vec), cmp_vec);
	}
}

/** Function to approximate the first derivative
 * 
 * @param desc pointer to the subgradient
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param info a pointer to a structu3re that contains the data and other information
 * @param idx vector of positions of the parameter to optimize
 * @param d number of coordinate in the block
 * @param objf a routine that computes the objective function
 */
static inline void afd(double *desc, double *param, int *len, void *info, int *idx, int *d, 
                       double (*objf)(double *, int *, void *)) {
    int i;
    double newf;
    double orgx;
    for (i = 0; i < *d; i++) {
        orgx = param[idx[i]];
	param[idx[i]] = orgx + STEP_DERIV;
	newf = objf(param, len, info);
	param[idx[i]] = orgx - STEP_DERIV;
	newf -= objf(param, len, info);
	param[idx[i]] = orgx;
        desc[idx[i]] = newf / (2.0 * STEP_DERIV);
    }
}

/** Function to approximate the second derivative FIXME: REQUIRES INVERSE HESSIAN HERE!!!
 * 
 * @param desc pointer to the subgradient
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param info a pointer to a structure that contains the data and other information
 * @param idx vector of positions of the parameter to optimize
 * @param d number of coordinate in the block
 * @param objf a routine that computes the objective function
 */
static void asd(double *desc, double *param, int *len, void *info, int *idx, int *d,
                double (*objf)(double *, int *, void *)) {
    int i;
    double newf;
    double orgx;
    for (i = 0; i < *d; i++) {
      orgx = param[idx[i]];
      param[idx[i]] = orgx + STEP_DERIV;
      newf = objf(param, len, info);
      param[idx[i]] = orgx - STEP_DERIV;
      newf += objf(param, len, info);
      param[idx[i]] = orgx;
      newf -= 2.0 * objf(param, len, info);
      newf /= (STEP_DERIV * STEP_DERIV);
      desc[i] = newf;
    }
}

/**
 * Adaptive d-block coordinate descent
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param objf a routine that computes the objective function
 */
void adaptive_dbcd(double *param, int *len, int *n_iter, void *info,
                   double (*objf)(double *, int *, void *)) {
    int t, i, k, d;
    unsigned np = *len;
    double newf, oldf, oldg = (double) INFINITY;
    double lr, th;
    double *desc = NULL;
    double *invh = NULL;
    int *idx = NULL;
    vec *grd = NULL;

    if (param && len && n_iter && objf) {
        if (*len > 0 && *n_iter > 0) {
	    idx = (int *) calloc(np, sizeof(int));
	    grd = (vec *) calloc(np, sizeof(vec));
	    desc = (double *) calloc(np, sizeof(double));
	    invh = (double *) calloc(np, sizeof(double));
	    if (idx && desc && invh && grd) {
		/* Compute the objective function */
		oldf = oldg;
		newf = objf(param, len, info);
		for (t = 0; t < *n_iter && newf < oldg; t++) {
		    oldg = newf;
		    update_grad(grd, param, len, info, objf);
		    /* Populate the block */
		    th = 0.25 * (grd[np - 1].v + grd[0].v);
		    th += 0.25 * (grd[np >> 1].v + grd[(np >> 1) - !(np & 1)].v);
		    for (d = i = 0; i < np; i++) {
			    idx[i] = grd[i].i;
			    d += (int) (grd[i].v > th);
		    }
                    for (k = 0; k < *n_iter && newf < oldf; k++) { /* wihtin block optim */
			lr = get_learning_rate((unsigned) k, (unsigned) *n_iter); 
                        oldf = newf;
                        afd(desc, param, len, info, idx, &d, objf);
			asd(invh, param, len, info, idx, &d, objf);
			for (i = 0; i < d; i++) {
				desc[i] /= 0.5 * EPSILON + fabs(0.5 * EPSILON + invh[i]);/**FIXME: needs a change*/
				param[idx[i]] -= lr * desc[i];
			}
			newf = objf(param, len, info);
		    }
		}
	    }
	    if (idx) free(idx);
	    if (grd) free(grd);
	    if (desc) free(desc);
	    if (invh) free(invh);
	}
    }
}

#ifdef DEBUG
double function(double *par, int *p, void *info) {
    double res;
    res = 0.5 + 0.5 * cos(par[1]);
    res /= 1.0 + exp(-par[0]);
    return 1.0 - res;
}

double function2(double *par, int *p, void *info) {
    double res;
    res = pow(fabs(par[1] * par[2] - 5.6789), 1.5);
    res /= 1.0 + sqrt(fabs(2.254 - par[0]) + 0.5);
    res += fabs(sin(par[2] + par[0])) / (1.0 + (exp(par[3] * par[0]) - exp(-par[3] * par[1])));
    return res;
}

int main() {
    double unk[4] = {-3.0, 1.0, 5.0, 0.5};
    int p = 2, maxit = 10000;
    int i;
    printf("Testing the min of 1.0 - (0.5 + 0.5 * cos(x)) * (0.5 + 0.5 * sin(y)):\n");
    printf("\t x = %g, y = %g\n", unk[0], unk[1]);
    printf("\t fun(x, y) = %g\n", function(unk, &p, 0));
    adaptive_dbcd(unk, &p, &maxit, &maxit, function);
    printf("Optimized results:\n");
    printf("\t x = %g, y = %g\n", unk[0], unk[1]);
    printf("\t fun(x, y) = %g\n\n", function(unk, &p, 0));
    p = 4;
    unk[0] = -3.0;
    unk[1] = 1.0;
    printf("Testing the min of function 2:\n");
    printf("\t ");
    i = 0;
    printf("x[%d] = %g", i, unk[i]);
    for (i++; i < p; i++) printf(", x[%d] = %g", i, unk[i]);
    printf("\n");
    printf("\t fun(x_vec) = %g\n", function2(unk, &p, 0));
    adaptive_dbcd(unk, &p, &maxit, &maxit, function2);
    printf("Optimized results:\n");
    printf("\t ");
    i = 0;
    printf("x[%d] = %g", i, unk[i]);
    for (i++; i < p; i++) printf(", x[%d] = %g", i, unk[i]);
    printf("\n");
    printf("\t fun(x_vec) = %g\n", function2(unk, &p, 0));
    return 0;
}
#endif

