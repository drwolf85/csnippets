#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5
#define STEP_DERIV 0.0125
#define LEARNING_RATE 0.9

typedef struct greedy_vec {
    double v;
    int i;
} vec_t;

int cmp_grd(const void *aa, const void *bb) {
    vec_t a = *(vec_t *) aa;
    vec_t b = *(vec_t *) bb;
    int res = (int) (fabs(a.v) < fabs(b.v)) * 2 - 1;
    return res;
}

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
 * Cyclic coordinate descent
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param objf a routine that computes the objective function
 */
void greedy_cd(double *param, int *len, int *n_iter, void *info,
          double (*objf)(double *, int *, void *)) {
    int t, i, k, np = *len;
    double newf, oldf = INFINITY;
    double desc;
    vec_t *grd;

    grd = (vec_t *) malloc(*len * sizeof(vec_t));

    if (grd) {
        /* Compute the objective function */
        newf = objf(param, len, info);
        k = -1;
        for (t = 0; t < *n_iter * np && newf < oldf; t++) {
            for (i = 0; i < np; i++) {
                grd[i].v = afd(param, len, info, i, objf);
                grd[i].i = i;
            }
            grd[k].v *=  1.0 - (double) (k >=0); 
            qsort(grd, np, sizeof(vec_t), cmp_grd);
            /* printf("First: %g - vs - Last: %f\n", grd[0].v, grd[np - 1].v); */
            i = grd[0].i * (int) (grd[0].i != k) + grd[1].i * (int) (grd[0].i == k);
            for (k = 0; k < *n_iter && newf < oldf; k++) {
                oldf = newf;
                desc = afd(param, len, info, i, objf);
                desc /= EPSILON + fabs(asd(param, len, info, i, objf));
                param[i] -= LEARNING_RATE * desc;
                newf = objf(param, len, info);
            }
            k = i;
        }
    }
    free(grd);
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
    greedy_cd(unk, &p, &maxit, 0, function);
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
    greedy_cd(unk, &p, &maxit, 0, function2);
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
