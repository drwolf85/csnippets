#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define LEARNING_RATE 0.001
#define BETA_1 0.9
#define BETA_2 0.999
#define EPSILON 1e-8
#define FACTOR_P 0.01 
/**
 * It computes the gradient of the objective function, updates the momentum and the second order
 * momentum, and then updates the parameters using the Lion Algorithm
 * 
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param grad a routine that computes the gradient of the objective function
 */
void lion(double *param, int *len, int *n_iter, void *info,
          void (*grad)(double *, double *, int *, void *)) {
    int t, i, np = *len;
    double *grd_v;
    double *mom_m;
    double *mom_c;
    double sgn;

    grd_v = (double *) malloc(np * sizeof(double));
    mom_c = (double *) malloc(np * sizeof(double));
    mom_m = (double *) calloc(np, sizeof(double));
    if (mom_m && mom_c && grd_v) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            #pragma omp parallel for simd private(sgn)
            for (i = 0; i < np; i++) {
                /* Lion update of custom momentum */
                mom_c[i] = BETA_1 * mom_m[i] + (1.0 - BETA_1) * grd_v[i];
                /* Update the momentum */
                mom_m[i] *= BETA_2;
                mom_m[i] +=  (1.0 - BETA_2) * grd_v[i];
                /* Lion update */
                sgn = (double) (mom_c[i] > 0.0) - (double) (mom_c[i] < 0.0);
                /* Computing the step */
                grd_v[i] = sgn + FACTOR_P * param[i];
                grd_v[i] *= LEARNING_RATE;
                param[i] -= grd_v[i];
            }
        }
    }
    free(mom_c);
    free(mom_m);
    free(grd_v);
}
