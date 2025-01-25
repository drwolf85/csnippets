#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5
#define STEP_DERIV 0.0125
#define LEARNING_RATE 0.03

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

/**
 * @brief Compute sorted gradient
 * 
 * @param grd Pointer to an array where to store the gradient values (sorted by their absolute values)
 * @param param Pointer to a vector of parameters
 * @param len Pointer to the number of parameters
 * @param info Pointer to potential data structures for computing the objective function and its gradient
 * @param invmaxlip Value of the inverse maximum Lipschitz coefficient
 * @param objf Pointer to the objective function
 */
static inline void sorted_grad(vec_t *grd, double *param, int *len, void *info, double invmaxlip,
                               double (*objf)(double *, int *, void *)) {
    int i, np = *len; 
    for (i = 0; i < np; i++) {
        grd[i].v = afd(param, len, info, i, objf) * invmaxlip;
        grd[i].i = i;
    }
    qsort(grd, np, sizeof(vec_t), cmp_grd);
}

/**
 * Coordinate descent
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param objf a routine that computes the objective function
 */
void coord_desc(double *param, int *len, int *n_iter, void *info,
          double (*objf)(double *, int *, void *)) {
    int t, i, np = *len;
    double newf, oldf = INFINITY;
    vec_t *grd;

    grd = (vec_t *) malloc(*len * sizeof(vec_t));

    if (grd) {
        newf = objf(param, len, info);
        sorted_grad(grd, param, len, info, 1.0, objf);
        double const desc = 1.0 / fabs(grd[0].v);
        for (t = 0; t < *n_iter * np; t++) {
            oldf = newf;
            i = grd[0].i;
            param[i] -= LEARNING_RATE * grd[0].v;
            newf = objf(param, len, info);
            if (newf >= oldf) break;
            sorted_grad(grd, param, len, info, desc, objf);
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
    coord_desc(unk, &p, &maxit, 0, function);
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
    coord_desc(unk, &p, &maxit, 0, function2);
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
