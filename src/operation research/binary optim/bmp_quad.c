#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define MAX_ITER 1000
#define EXP_ITER_DEN 33

/**
 * @brief Binary Mathematical Programming for binary quadratic forms 
 * 
 * @param x Pointer to array of `{-1, 1}`
 * @param n Number of elements in the array
 * @param syx Pointer to a vector (of size `n`)
 * @param Sxx  Pointer to a square matrix (of size `n x n`)
 * @param sz_step Size of the gradient descent step
 * @param grad Pointer to gradient function
 */
void bmp_bin_quad(double *x, uint32_t n, double *syx, double *Sxx, double sz_step,
                 void (*grad)(double *, double *, uint32_t, double *, double *)) {
    uint32_t i, t = 1;
    double sr, inprd;
    double *g = calloc(n, sizeof(double));

    if (x && syx && Sxx && g) {
        memset(x, 0, sizeof(double) * n); /* Initialize values of x*/
        while (t < MAX_ITER) {
            inprd = sz_step * (double) t / (double) EXP_ITER_DEN;
            grad(g, x, n, syx, Sxx); /* Compute gradient */
            for (i = 0; i < n; i++) {
                sr = x[i] * x[i];
                g[i] *= 1.0 - sr;
                x[i] = tanh(atanh(x[i]) - inprd * g[i]); /* Update values */
            }
            t++;
        }
    }
    free(g);
}

#ifdef DEBUG

#define N 4
double mat[] = {1.0,  0.2,  0.3, -0.4, 
                0.2,  1.0, -0.5,  0.6,
                0.3, -0.5,  1.0, -0.7,
                -.4,  0.6, -0.7,  1.0};
double vec[] = {0.9, -0.75, 0.0, 0.2};

/**
 * @brief Objective Function
 * 
 * @param x Pointer to array of `{-1, 1}`
 * @param n Number of elements in the array
 * @param syx Pointer to a vector (of size `n`)
 * @param Sxx  Pointer to a square matrix (of size `n x n`)
 * @return double 
 */
double obj_fun(double *x, uint32_t n, double *syx, double *Sxx, bool bin) {
    uint32_t i, j;
    double res = 0.0;
    if (bin) for (i = 0; i < n; i++) x[i] = 2.0 * (double) (x[i] >= 0.0) - 1.0;
    for (i = 0; i < n; i++) {
        res -= x[i] * fabs(syx[i]);
        for (j = 0; j < n; j++) {
            res += 0.5 * x[i] * fabs(Sxx[i * n + j]) * x[j];
        }
    }
    return res;
}

/**
 * @brief Gradient of the Objective Function
 * 
 * @param g Pointer to an empty array where to store the gradient
 * @param x Pointer to array of `{-1, 1}`
 * @param n Number of elements in the array
 * @param syx Pointer to a vector (of size `n`)
 * @param Sxx  Pointer to a square matrix (of size `n x n`)
 * @return double 
 */
void my_grad(double *g, double *x, uint32_t n, double *syx, double *Sxx) {
    uint32_t i, j;
    for (i = 0; i < n; i++) {
        g[i] = -fabs(syx[i]);
        for (j = 0; j < n; j++) {
            g[i] += 0.25 * fabs(Sxx[i * n + j]) * x[j];
        }
    }
}

int main() {
    uint32_t i;
    double x[N];
    double o;
    bmp_bin_quad(x, N, vec, mat, 1.0, my_grad);
    o = obj_fun(x, N, vec, mat, false);
    printf("Approximate solution: ");
    for (i = 0; i < N; i++) printf("%.3f ", x[i]);
    printf("\n");
    printf("Objective function: %.6f\n", o);
    o = obj_fun(x, N, vec, mat, true);
    printf("Binary solution: ");
    for (i = 0; i < N; i++) printf("%1.0f ", x[i]);
    printf("\n");
    printf("Objective function: %.6f\n", o);
    return 0;
}
#endif
