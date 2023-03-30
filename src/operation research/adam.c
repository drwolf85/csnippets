#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define LEARNING_RATE 0.001
#define BETA_1 0.9
#define BETA_2 0.999
#define EPSILON 1e-8

/**
 * It computes the gradient of the objective function, updates the momentum and the second order
 * momentum, and then updates the parameters using the Adam algorithm
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param grad a pointer to a routine that computes the gradient of the objective function
 */
void adam(double *param, int *len, int *n_iter, void *info,
          void (*grad)(double *, double *, int *, void *)) {
    int t, i, np = *len;
    double sc_m, sc_s;
    double *grd_v;
    double *mom_m;
    double *mom_s;

    grd_v = (double *) malloc(np * sizeof(double));
    mom_m = (double *) calloc(np, sizeof(double));
    mom_s = (double *) calloc(np, sizeof(double));
    if (mom_m && mom_s && grd_v) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            /* Scaling factors */
            sc_m = 1.0 / (1.0 - pow(BETA_1, (double) t));
            sc_s = 1.0 / (1.0 - pow(BETA_2, (double) t));
            #pragma omp parallel for simd
            for (i = 0; i < np; i++) {
                /* Update the momentum */
                mom_m[i] *= BETA_1;
                mom_m[i] +=  (1.0 - BETA_1) * grd_v[i];
                mom_m[i] *= sc_m;
                /* Update second order momentum */
                mom_s[i] *= BETA_2;
                mom_s[i] +=  (1.0 - BETA_2) * grd_v[i] * grd_v[i];
                mom_s[i] *= sc_s;
                /* Computing the step */
                grd_v[i] = LEARNING_RATE * mom_m[i] / (sqrt(mom_s[i]) + EPSILON);
                param[i] -= grd_v[i];
            }
        }
    }
    free(mom_s);
    free(mom_m);
    free(grd_v);
}
