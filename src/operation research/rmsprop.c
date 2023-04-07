#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define LEARNING_RATE 0.001
#define GAMMA 0.999
#define EPSILON 1e-6

/**
 * It computes the gradient of the objective function, updates the second order
 * momentum, and then updates the parameters using the RMSProp algorithm
 *
 * @param param the parameters to be optimized
 * @param len the length of the parameter vector
 * @param n_iter number of iterations
 * @param info a pointer to a structure that contains the data and other information
 * @param grad a routine that computes the gradient of the objective function
 */
void rmsprop(double *param, int *len, int *n_iter, void *info,
<<<<<<< HEAD
          void (grad)(double *, double *, int *, void *)) {
=======
          void (*grad)(double *, double *, int *, void *)) {
>>>>>>> 8d45c9bd26212037b9abfe746a1cd799d0554e09
    int t, i, np = *len;
    double *grd_v;
    double *mom_s;

    grd_v = (double *) malloc(np * sizeof(double));
    mom_s = (double *) calloc(np, sizeof(double));
    if (mom_s && grd_v) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            /* Scaling factors */
            #pragma omp parallel for simd
            for (i = 0; i < np; i++) {
                /* Update second order momentum */
                mom_s[i] *= GAMMA;
                mom_s[i] +=  (1.0 - GAMMA) * grd_v[i] * grd_v[i];
                /* Computing the step */
                grd_v[i] *= LEARNING_RATE / (sqrt(mom_s[i] + EPSILON));
                param[i] -= grd_v[i];
            }
        }
    }
    free(mom_s);
    free(grd_v);
}
