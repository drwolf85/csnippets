#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas64.h>
#include <lapacke.h>

/* gcc ram.c -Os -march=native -shared -fpic -lm -llapacke -lblas -o ram.so */

double *W1, *bs; /* Randomized coeff. (L x h), (h)*/
double *W2; /* Trainable matrix [(L + h) x C] */
int L = 0, h = 0, C = 0;

#ifdef DEBUG
void print_matrix(double *mat, int nr, int nc) {
    int i, j;
    for (i = 0; i < nr; i++) {
        for (j = 0; j < nc; j++)
            printf("%.5f ", mat[nr * j + i]);
        printf("\n");
    }
}
#endif

static inline double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -30) - 1.0;
   s = a * a;
   b = ldexp((double) v, -30) - 1.0;
   s += b * b * (1.0 - s);
   s = -2.0 * log(s) / s;
   a = b * sqrtf(s);
   return mu + sd * a;
}

/****************************************************************
 *                    Random Additive Model 
 * 
 * This model is based on the random vector functional-link net 
 * proposed by Pao, Park, and Sobajic (1994)
 * 
 ****************************************************************/

extern void init_ram(int *dimX, int *nh, int *dimY) {
    int i, j;
    double const isqh = 1.0 / sqrt((double) *nh);
    double const isqL = 1.0 / sqrt((double) dimX[1]);
    C = dimY[1];
    L = dimX[1]; 
    h = *nh;
    bs = (double *) malloc(h * sizeof(double));
    W1 = (double *) malloc(L * h * sizeof(double));
    W2 = (double *) calloc((L + h) * C, sizeof(double));
    if (bs && W1) {
        srand(time(NULL));
        for (i = 0; i < h; i++) bs[i] = rnorm(0.0, isqh);
        for (i = 0; i < h * L; i++) W1[i] = rnorm(0.0, isqL);
    }
}

extern void free_ram() {
    C = 0;
    h = 0;
    L = 0;
    free(bs);
    free(W1);
    free(W2);
}

extern void fit_ram(double *Y, int *dimY, double *X, int *dimX, double (*acti)(double), double *reg_param, int *nh) {
    int i, j, Lph;
    int const m = dimX[0];
    int *ipiv;
    double lambda = 0.8;
    double *H, *Tmp, *mat;
    if (reg_param) if (*reg_param > 1e-9) lambda = 1.0 / *reg_param;
    if (X && dimX && Y && dimY) {
        if (dimX[1] > 0 && m > 0 && *dimX == *dimY) { 
            L = dimX[1];
            if (!(W1 && bs && W2) && (h > 0 || nh)) init_ram(dimX, h > 0 ? &h : nh, dimY);
            if (W1 && bs && W2) {
                if (m > 0 && h > 0 && L > 0 && C > 0) {
                    Lph = L + h;
                    H = (double *) calloc(m * Lph, sizeof(double));
                    Tmp = (double *) calloc(m < Lph ? m * m : Lph * Lph, sizeof(double));
                    ipiv = (int *) calloc(m < Lph ? m : Lph, 2 * sizeof(int));
                    if (H && Tmp && ipiv) {
                        memcpy(&H[m * h], X, m * L * sizeof(double)); /* copy data in H */
                        for (j = 0; j < h; j++) for (i = 0; i < m; i++) H[m * j + i] = bs[j]; /* Copy biases in H */
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, h, L, 1.0, X, m, W1, L, 1.0, H, m); /* Compute hidden layer */
                        if (acti) for (i = 0; i < m * Lph; i++) H[i] = acti(H[i]); /* Activation function */
                        memset(W2, 0, sizeof(double) * Lph * C); /* Reset values of the trainable matrix */
                        if (m >= Lph) {
                            mat = (double *) malloc(Lph * C * sizeof(double));
                            if (mat) {
                                for (i = 0; i < Lph; i++) Tmp[i * (Lph + 1)] = lambda;
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Lph, C, m, 1.0, H, m, Y, m, 0.0, mat, Lph);
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Lph, Lph, m, 1.0, H, m, H, m, 1.0, Tmp, Lph);
                                LAPACKE_dgetrf(LAPACK_COL_MAJOR, Lph, Lph, Tmp, Lph, ipiv);
                                LAPACKE_dgetri(LAPACK_COL_MAJOR, Lph, Tmp, Lph, ipiv);
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Lph, C, Lph, 1.0, Tmp, Lph, mat, Lph, 0.0, W2, Lph);
                            }
                            free(mat);
                        }
                        else {
                            mat = (double *) malloc(Lph * m * sizeof(double));
                            if (mat) {
                                for (i = 0; i < m; i++) Tmp[i * (m + 1)] = lambda;
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, Lph, 1.0, H, m, H, m, 1.0, Tmp, m);
                                LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, m, Tmp, m, ipiv);
                                LAPACKE_dgetri(LAPACK_COL_MAJOR, m, Tmp, m, ipiv);
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Lph, m, m, 1.0, H, m, Tmp, m, 0.0, mat, Lph);//
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Lph, C, m, 1.0, mat, Lph, Y, m, 0.0, W2, Lph);
                            }
                            free(mat);
                        }
                    }
                    free(H);
                    free(Tmp);
                    free(ipiv);
                }
            }
            else {
                free_ram();
            }
        }
    }
}

extern void save_ram(char *filename) {
    FILE *fp;
    if (C > 0 && L > 0 && h > 0) {
        if (W1 && bs && W2) {
            fp = fopen(filename, "wb");
            if (fp) {
                fwrite(&h, sizeof(int), 1, fp);
                fwrite(bs, sizeof(double), h, fp);
                fwrite(&L, sizeof(int), 1, fp);
                fwrite(W1, sizeof(double), L * h, fp);
                fwrite(&C, sizeof(int), 1, fp);
                fwrite(W2, sizeof(double), (L + h) * C, fp);
                fclose(fp);
            }
        }
    }
}

extern void load_ram(char *filename) {
    FILE *fp;
    if (C > 0 && L > 0 && h > 0) {
        if (W1 && bs && W2) {
            fp = fopen(filename, "rb");
            if (fp) {
                fread(&h, sizeof(int), 1, fp);
                bs = (double *) malloc(h * sizeof(double));
                fread(bs, sizeof(double), h, fp);
                fread(&L, sizeof(int), 1, fp);
                W1 = (double *) malloc(L * h * sizeof(double));
                fread(W1, sizeof(double), L * h, fp);
                fread(&C, sizeof(int), 1, fp);
                W2 = (double *) malloc((L + h) * C * sizeof(double));
                fread(W2, sizeof(double), (L + h) * C, fp);
                fclose(fp);
            }
        }
    }
}

extern double * eval_ram(double *X, int *dimX, double (*acti)(double)) {
    int i, j;
    int const m = dimX[0];
    double *H;
    double *R;
    if (X && dimX && W1 && bs && W2) {
        if (m > 0 && h > 0 && L > 0 && C > 0) {
            R = (double *) calloc(m * C, sizeof(double));
            H = (double *) calloc(m * (L + h), sizeof(double));
            if (R && H) {
                memcpy(&H[m * h], X, m * L * sizeof(double)); /* copy data in H */
                for (j = 0; j < h; j++) for (i = 0; i < m; i++) H[m * j + i] = bs[j]; /* Copy biases in H */
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, h, L, 1.0, X, m, W1, L, 1.0, H, m); /* Compute hidden layer */
                if (acti) for (i = 0; i < m * (L + h); i++) H[i] = acti(H[i]); /* Activation function */
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, C, L + h, 1.0, H, m, W2, L + h, 0.0, R, m); /* Compute output layer */
            }
            free(H);
        }
    }
    return R;
}

#ifdef DEBUG
double my_soft_sign(double x) {
    return x / (fabs(x) + 1.0);
}

int main (int nargc, char **args) {
    int i, j;
    double x[] = {-1.0, 1.0, 0.0, 0.5, -0.5, \
                   1.2, 2.1, -1.2, 0.4, -0.3, \
                  -2.1, 0.7, 0.8, -0.4, -1.6};
    int dx[2] = {5, 3};
    double y[] = {-0.01, 0.02, 0.03, 0.04, -0.05, \
                  1.0, 2.0, 0.1, 0.2, 0.3};
    double *res;
    int dy[2] = {5, 2};
    int dh = 2;
    double regularization = 18.0;
    char fn[] = {"test.bin"};
    init_ram(dx, &dh, dy);
    fit_ram(y, dy, x, dx, my_soft_sign, &regularization, 0);
    res = eval_ram(x, dx, my_soft_sign);
    if (res) {
        printf("Fitted values:\n");
        print_matrix(res, 5, 2);
    }
    free(res);
    if (nargc > 1) {
        save_ram(args[1]);
    }
    free_ram();
    return 0;
}
#endif
