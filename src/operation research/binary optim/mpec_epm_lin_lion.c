#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_RHO 0.8
#define DEFAULT_SIGMA 1.01
#define MAX_ITER 100
#define MY_EPS_TOLL 1e-9


#define INNER_MAX_ITER 20
#define LEARNING_RATE 0.001
#define BETA_1 0.9
#define BETA_2 0.999
#define FACTOR_P 0.01

/**
 * @brief Mathematical Programming with Equilibrium Constraints - Exact Penalty Method
 * 
 * @param x A pointer to a vector of unknown values to optimize in {-1, 1}^n
 * @param c A pointer to a known vector of cost (use to compute a linear objective function)
 * @param n Length of the vector `x`
 * @param A A pointer to a matrix (pxn) of contraint sums
 * @param b A pointer to a vector of constraint totals
 * @param p Number of linear constraints
 * @param rho Penalty parameter (like a Lagrangian multiplier)
 * @param sigma Scaling parameter operating on `rho`
 * @param T Number of void iterations on the updates of `rho` 
 */
void mpec_epm_lin_lion(double *x, double *c, size_t n, double *A, double *b, size_t p, double rho, double sigma, size_t T) {
    size_t i, j, l, t = 1;
    double sr, trx, sgn, inprd;
    double L = fabs(*c);

    double *e = NULL;
    double *v = (double *) calloc(n, sizeof(double));
    double *g = (double *) malloc(n * sizeof(double));
    double *mom_m = (double *) calloc(n, sizeof(double));

    rho = (double)(rho > 0.0) * rho + (double)(rho <= 0.0) * DEFAULT_RHO;
    sigma = (double)(sigma > 1.0) * sigma + (double)(sigma <= 1.0) * DEFAULT_SIGMA;
    for (i = 1; i < n; i++) L += (double) (fabs(c[i]) > L) * (fabs(c[i]) - L);
    L *= 2.0;

    if (A && b && p > 0) e = (double *) malloc(p * sizeof(double));
    if (mom_m && x && v && g && n > 0) {
        memset(x, 0, sizeof(double) * n); /* Initialize values of x*/
        while(t <= MAX_ITER) { /** FIXME: there are other conditions to consider...*/
            /* Primal step */
            /* c^T x + rho * (n - x^T v) : x \in [-1, 1] \cap constraint set... which is equivalent to*/
            inprd = (double) t / (double) T;
            if (e) {                
                for (l = 0; l < INNER_MAX_ITER; l++) {
                    for (j  = 0; j < p; j++) {
                    /* constraint: A %*% x <= b */
                        e[j] = 0.0;
                        for (i = 0; i < n; i++) {
                        e[j] += A[n * j + i] * x[i]; /** NOTE: this is column-major format */                       
                        }
                        e[j] -= b[j];
                        e[j] *= L * (double) (e[j] > 0.0);
                    }
                    /* (c - rho * v)^T x + rho * n : x \in [-1, 1] \cap constraint set...*/
                    for (i = 0; i < n; i++) {
                        sr = x[i] * x[i];
                        g[i] = 0.0;
                        for (j  = 0; j < p; j++) {
                            g[i] += A[n * j + i] * e[j]; /** NOTE: this is column-major format */ 
                        }
                        g[i] += c[i] - rho * v[i];
                        g[i] *= 1.0 - sr;
                        /* Begin Lion Step */
                        trx = atanh(x[i]);
                        sgn = BETA_1 * mom_m[i] + (1.0 - BETA_1) * g[i];
                        mom_m[i] *= BETA_2;
                        mom_m[i] +=  (1.0 - BETA_2) * g[i];
                        sgn = (double) (sgn > 0.0) - (double) (sgn < 0.0);
                        g[i] = sgn + FACTOR_P * trx;
                        g[i] *= LEARNING_RATE * inprd;
                        trx -= g[i];
                        /* End of Lion Step */
                        x[i] = tanh(trx);
                    }
                }
            } else {
                for (l = 0; l < INNER_MAX_ITER; l++) {
                    /* (c - rho * v)^T x + rho * n : x \in [-1, 1] \cap constraint set...*/
                    for (i = 0; i < n; i++) {
                        sr = x[i] * x[i];
                        g[i] = c[i] - rho * v[i];
                        g[i] *= 1.0 - sr;
                        /* Begin Lion Step */
                        trx = atanh(x[i]);
                        sgn = BETA_1 * mom_m[i] + (1.0 - BETA_1) * g[i];
                        mom_m[i] *= BETA_2;
                        mom_m[i] +=  (1.0 - BETA_2) * g[i];
                        sgn = (double) (sgn > 0.0) - (double) (sgn < 0.0);
                        g[i] = sgn + FACTOR_P * trx;
                        g[i] *= LEARNING_RATE * inprd;
                        trx -= g[i];
                        /* End of Lion Step */
                        x[i] = tanh(trx);
                    }
                }
            }
            /* Dual step */
            inprd = 0.0;
            for (i = 0; i < n; i++)
                inprd += x[i] * x[i];
            if (fabs(inprd) > MY_EPS_TOLL) {
                inprd = sqrt((double) n / inprd);
                for (i = 0; i < n; i++)
                    v[i] = x[i] * inprd;
            }
            /* Updating penalty parameter */
            sr = sigma * rho;
            rho += (double) !(t % T) * ((L + (double)(L > sr) * (sr - L)) - rho);
            t++;
        }
    }
    free(v);
    free(g);
    free(e);
    free(mom_m);
}

/* Test function */

#ifdef DEBUG

#include <stdio.h>
#define N 5

void main() {
    double x[N] = {0};
    double c[] = {-3.0, 1.0, -0.2, 1.2, 3.0};
    double A[] = {1.0, 1.0, 0.0, 0.0, 0.0, \
                  0.0, 0.0, 1.0, 1.0, 1.0};
    double b[] = {0.01, -1.9};
    // double b[] = {0.01, -0.9};
    int const p = 2;
    int i;

    printf("Binary optim without linear constraints:\n");
    mpec_epm_lin_lion(x, c, N, 0, 0, 0, -1.0, -1.0, 15);
    for (i = 0; i < N; i++)
        printf("x[%d] = %f -> %d\n", i, x[i], (int) (x[i] >= 0.0));

    printf("\nBinary optim with linear constraints:\n");
    mpec_epm_lin_lion(x, c, N, A, b, p, -1.0, -1.0, 15);
    for (i = 0; i < N; i++)
        printf("x[%d] = %f -> %d\n", i, x[i], (int) (x[i] >= 0.0));
}

#endif

