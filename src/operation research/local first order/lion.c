#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define LEARNING_RATE 0.001
#define BETA_1 0.9
#define BETA_2 0.999
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
    double sgn;

    grd_v = (double *) malloc(np * sizeof(double));
    mom_m = (double *) calloc(np, sizeof(double));
    if (mom_m && grd_v) {
        for (t = 0; t < *n_iter; t++) {
            /* Update the gradient */
            (*grad)(grd_v, param, len, info);
            #pragma omp parallel for simd private(sgn)
            for (i = 0; i < np; i++) {
                /* Lion update of custom momentum */
                sgn = BETA_1 * mom_m[i] + (1.0 - BETA_1) * grd_v[i];
                /* Update the momentum */
                mom_m[i] *= BETA_2;
                mom_m[i] +=  (1.0 - BETA_2) * grd_v[i];
                /* Lion update */
                sgn = (double) (sgn > 0.0) - (double) (sgn < 0.0);
                /* Computing the step */
                grd_v[i] = sgn + FACTOR_P * param[i];
                grd_v[i] *= LEARNING_RATE;
                param[i] -= grd_v[i];
            }
        }
    }
    free(mom_m);
    free(grd_v);
}

#ifdef DEBUG

double obj(double *B_col_maj, int p, double *Y_row_maj, int n) {
    double res = 0.0;
    double tmp;
    int i, j, k;
    #pragma omp parallel for simd private(i, j, k, tmp) reduction(+ : res) collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            tmp = Y_row_maj[p * i] * B_col_maj[p * j];
            for (k = 1; k < p; k++) {
                tmp += Y_row_maj[p * i + k] * B_col_maj[p * j + k];
            }
            res += fabs(tmp - Y_row_maj[p * i + j]);
        }
    }
    return res;
}

struct data_grad {
    int n;
    int p;
    double *y;
    int t;
};

#include <stdio.h>
#include <string.h>
#include <time.h>

void obj_grad(double *g, double *B, int *p2, void *info) {
    struct data_grad * mat = (struct data_grad *) info;
    int i, j, k, pos;
    double res = 0.0;
    double tmp;
    double *smat = (double *) calloc(mat->p * mat->n, sizeof(double));
    if ((mat->t + 1) % 1000 == 0) printf("Objective: %g\n", obj(B, mat->p, mat->y, mat->n));
    mat->t++;
    memset(g, 0, *p2 * sizeof(double));
    if (smat) {
        #pragma omp parallel for simd private(i, j, k, tmp) collapse(2)
        for (i = 0; i < mat->n; i++) {
            for (j = 0; j < mat->p; j++) {
                tmp = 0.0;
                for (k = 0; k < mat->p; k++) {
                    tmp += mat->y[mat->p * i + k] * (B[mat->p * j + k] - (double) (j == k));
                }
                smat[i + j * mat->n] = (double) (tmp > 0.0) - (double) (tmp < 0.0);
            }
        }
        #pragma omp parallel for simd private(i, j, k, tmp) collapse(2)
        for (j = 0; j < mat->p; j++) {
            for (k = 0; k < mat->p; k++) {
                tmp = 0.0;
                for (i = 0; i < mat->n; i++) {
                    tmp += smat[j * mat->n + i] * mat->y[mat->p * i + k];
                }
                g[j * mat->p + k] = tmp;
            }
        }
    }
    free(smat);
    #pragma omp parallel for simd private(i, pos)
    for (i = 0; i < mat->p; i++) {
        pos = i * (mat->p + 1);
        B[pos] = g[pos] = 0.0;
    }
}

int main() {
    double B[25] = {0};
    double y[] = {-0.1315819, 0.1793963, -0.01755624, 0.3562621, 0.4973845, 0.2482045, 0.4099135, 0.2161882, -0.4980195, -0.2372604, 0.230425, -0.1300743, 0.8188969, -0.01759944, 0.2121086, 0.8206889, 0.3538281, 0.5556703, 0.2935429, 0.3055268, -0.2001838, -0.6887512, -0.6138871, -0.7589452, -0.3999022, 0.3259763, 0.9863215, 0.1321867, 1.363693, 0.7105176, -0.4082712, -0.7169683, -0.6221118, -0.764756, -0.3655938, -0.6868594, -0.1536153, -1.051096, -0.3658161, -1.200048, -1.50248, -1.335725, -1.717085, 0.1390045, -0.9140004, -1.488441, -1.692978, -1.967536, -1.009726, -0.6429252};
    struct data_grad myinfo = {n: 10, p: 5, y: y, t: 0};
    int max_it = 5000;
    int par_len = 25;
    int i;
    lion(B, &par_len, &max_it, &myinfo, obj_grad);
    for (i = 0; i < par_len; i++)
        printf("%s%.8f%s", B[i], B[i] < 0.0 ? "" : " ", (i + 1) % 5 == 0 ? "\n" : " ");
    return 0;
}
#endif
