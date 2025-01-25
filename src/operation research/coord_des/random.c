#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5
#define STEP_DERIV 0.0125
#define LEARNING_RATE 0.9

/** Function to approximate the first derivative
 * 
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param info a pointer to a structure that contains the data and other information
 * @param i position number of the parameter to optimize within the array `param`
 * @param objf a routine that computes the objective function
 */
static inline double afd(double *param, int *len, void *info, int i, 
                         double (*objf)(double *, int *, void *)) {
    double orgx = param[i];
    double newf;
    param[i] = orgx + STEP_DERIV;
    newf = objf(param, len, info);
    param[i] = orgx - STEP_DERIV;
    newf -= objf(param, len, info);
    param[i] = orgx;
    return newf / (2.0 * STEP_DERIV);
}

/** Function to approximate the second derivative
 * 
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param info a pointer to a structure that contains the data and other information
 * @param i position number of the parameter to optimize within the array `param`
 * @param objf a routine that computes the objective function
 */
double asd(double *param, int *len, void *info, int i, 
           double (*objf)(double *, int *, void *)) {
    double orgx = param[i];
    double newf;
    param[i] = orgx + STEP_DERIV;
    newf = objf(param, len, info);
    param[i] = orgx - STEP_DERIV;
    newf += objf(param, len, info);
    param[i] = orgx;
    newf -= 2.0 * objf(param, len, info);
    newf /= (STEP_DERIV * STEP_DERIV);
    return newf;
}

/**
 * Random coordinate descent
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param objf a routine that computes the objective function
 */
void random_cd(double *param, int *len, int *n_iter, void *info,
          double (*objf)(double *, int *, void *)) {
    int t, i, k, np = *len;
    int *idx;
    double newf, oldf = INFINITY;
    double desc;

    idx = (int *) malloc(*len * sizeof(int));
    /* Compute the objective function */
    if (idx) {
        newf = objf(param, len, info);
        for (t = 0; t < *n_iter && newf < oldf; t++) {
            for (i = 0; i < np; i++) idx[i] = i; /* Populate vector of indices */
            for (i = 0; i < np; i++) {
                k = rand() % np;
                if (k != i) {
                    idx[i] ^= idx[k]; 
                    idx[k] ^= idx[i]; 
                    idx[i] ^= idx[k];
                }
            }
            for (i = 0; i < np; i++) {
                for (k = 0; k < *n_iter && newf < oldf; k++) {
                    oldf = newf;
                    desc = afd(param, len, info, idx[i], objf);
                    desc /= EPSILON + fabs(asd(param, len, info, idx[i], objf));
                    param[idx[i]] -= LEARNING_RATE * desc;
                    newf = objf(param, len, info);
                }
            }
        }
    }
    free(idx);
}

#ifdef DEBUG
double function2(double *par, int *p, void *info) {
    double res;
    res = pow(fabs(par[1] * par[2] - 5.6789), 1.5);
    res /= 1.0 + sqrt(fabs(2.254 - par[0]) + 0.5);
    res += fabs(sin(par[2] + par[0])) / (1.0 + (exp(par[3] * par[0]) - exp(-par[3] * par[1])));
    return res;
}

int main() {
    double unk[4] = {-3.0, 1.0, 5.0, 0.5};
    int p = 4, maxit = 10000;
    int i;
    srand(time(NULL));
    printf("Testing the min of a function:\n");
    printf("\t ");
    i = 0;
    printf("x[%d] = %g", i, unk[i]);
    for (i++; i < p; i++) printf(", x[%d] = %g", i, unk[i]);
    printf("\n");
    printf("\t fun(x_vec) = %g\n", function2(unk, &p, 0));
    random_cd(unk, &p, &maxit, 0, function2);
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
