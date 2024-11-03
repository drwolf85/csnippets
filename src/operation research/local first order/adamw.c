#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define LEARNING_RATE 0.001
#define BETA_1 0.9
#define BETA_2 0.999
#define EPSILON 1e-8
#define FACTOR_P 0.01 
#define FACTOR_LR 0.001

/**
 * It computes the gradient of the objective function, updates the momentum and the second order
 * momentum, and then updates the parameters using the AdamW algorithm
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param grad a routine that computes the gradient of the objective function
 */
void adamw(double *param, int *len, int *n_iter, void *info,
          void (*grad)(double *, double *, int *, void *)) {
    int t, i, np = *len;
    double *grd_v;
    double *mom_m;
    double *mom_s;
    double sc_m, sc_s, pbm = 1.0, pbs = 1.0;

    grd_v = (double *) malloc(np * sizeof(double));
    mom_m = (double *) calloc(np, sizeof(double));
    mom_s = (double *) calloc(np, sizeof(double));
    if (mom_m && mom_s && grd_v) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            /* Scaling factors */
            pbm *= BETA_1;
            pbs *= BETA_2;
            sc_m = 1.0 / (1.0 - pbm);
            sc_s = 1.0 / (1.0 - pbs);
            #pragma omp parallel for simd
            for (i = 0; i < np; i++) {
                /* Update the momentum */
                mom_m[i] *= BETA_1;
                mom_m[i] +=  (1.0 - BETA_1) * grd_v[i];
                /* Update second order momentum */
                mom_s[i] *= BETA_2;
                mom_s[i] +=  (1.0 - BETA_2) * grd_v[i] * grd_v[i];
                /* Computing the step */
                grd_v[i] = mom_m[i] * sc_m * (sqrt(mom_s[i] * sc_s) + EPSILON);
                grd_v[i] += param[i] * FACTOR_P;
                grd_v[i] *= LEARNING_RATE * FACTOR_LR;
                param[i] -= grd_v[i];
            }
        }
    }
    free(mom_s);
    free(mom_m);
    free(grd_v);
}
