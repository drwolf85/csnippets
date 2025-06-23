#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5
#define STEP_DERIV 0.0125
#define LEARNING_RATE 0.9

/** Function to approximate the first derivative
 * 
 * @param desc pointer to the subgradient
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param info a pointer to a structure that contains the data and other information
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
static inline void asd(double *desc, double *param, int *len, void *info, int *idx, int *d,
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
 * Random d-block coordinate descent
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param d number of coordinates forming a block
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param objf a routine that computes the objective function
 */
void random_dbcd(double *param, int *len, int *d, int *n_iter, void *info,
          double (*objf)(double *, int *, void *)) {
    int t, i, k, j;
    unsigned np = *len;
    double newf, oldf, oldg = (double) INFINITY;
    double *desc = NULL;
    double *invh = NULL;
    int *idx = NULL;

    if (param && len && d && n_iter && info && objf) {
        if (*d > 0 && *len > 0 && *n_iter > 0) {
	    *d = np >= (unsigned) *d ? *d : (int) np;
	    idx = (int *) calloc(np, sizeof(int));
	    desc = (double *) calloc(np, sizeof(double));
	    invh = (double *) calloc(np, sizeof(double));
	    if (idx && desc && invh) {
		/* Populate the vector of parameter indices */
		for (i = 0; i < np; i++) idx[i] = i;
		/* Compute the objective function */
		oldf = oldg; 
		newf = objf(param, len, info);
		for (t = 0; t < *n_iter && newf < oldg; t++) {
		    oldg = newf;
		    /* Randomize the block by shuffling `*d` entries */
		    for (i = 0; i < *d; i++) {
			    k = (int) (arc4random() % (uint32_t) np);
			    j = idx[i];
			    idx[i] = idx[k];
			    idx[k] = j;
		    }
		    /* Wihtin block optim */
                    for (k = 0; k < *n_iter && newf < oldf; k++) {
                        oldf = newf;
                        afd(desc, param, len, info, idx, d, objf);
			asd(invh, param, len, info, idx, d, objf);
			for (i = 0; i < *d; i++) {
				desc[i] /= 0.5 * EPSILON + fabs(0.5 * EPSILON + invh[i]);/**FIXME: needs a change*/
				param[idx[i]] -= LEARNING_RATE * desc[i];
			}
			newf = objf(param, len, info);
		    }
		}
	    }
	    if (idx) free(idx);
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
    int p = 2, d = 2, maxit = 10000;
    int i;
    printf("Testing the min of 1.0 - (0.5 + 0.5 * cos(x)) * (0.5 + 0.5 * sin(y)):\n");
    printf("\t x = %g, y = %g\n", unk[0], unk[1]);
    printf("\t fun(x, y) = %g\n", function(unk, &p, 0));
    random_dbcd(unk, &p, &d, &maxit, &maxit, function);
    printf("Optimized results:\n");
    printf("\t x = %g, y = %g\n", unk[0], unk[1]);
    printf("\t fun(x, y) = %g\n\n", function(unk, &p, 0));
    p = 4;
    d = 2;
    unk[0] = -3.0;
    unk[1] = 1.0;
    printf("Testing the min of function 2:\n");
    printf("\t ");
    i = 0;
    printf("x[%d] = %g", i, unk[i]);
    for (i++; i < p; i++) printf(", x[%d] = %g", i, unk[i]);
    printf("\n");
    printf("\t fun(x_vec) = %g\n", function2(unk, &p, 0));
    random_dbcd(unk, &p, &d, &maxit, &maxit, function2);
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
