#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
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
static void lion(double *param, int *len, int *n_iter, void *info,
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

static inline double logis(double x) {
	return 1.0 / (1 + exp(- x));
}

double nll_contrib(double *bc, double *br, double *ac, double *ar, /* Parameters */
                       uint8_t yc, double *xc, size_t dc, /* Data Ca. */
                       uint8_t yr, double *xr, size_t dr) { /* Data Reca. */
        if (!(bc && br && ac && ar && xc && xr)) return nan("");
	double pc = *ac;
	double pr = *ar;
	double res = 0.0;
	double K;
	size_t j;
	for (j = 0; j < dc; j++) pc += xc[j] * bc[j];
	for (j = 0; j < dr; j++) pr += xr[j] * br[j];
	pc = logis(pc);
	pr = logis(pr);
	K = 1.0 - pc;
	K *= 1.0 - pr;
	K = 1.0 - K;
	res  = (double) (yc > 0) * (log(pc) + (double) (yr == 0) * log1p(-pr));
	res += (double) (yr > 0) * (log(pr) + (double) (yc == 0) * log1p(-pc));
	res -= (double) (yr > 0 || yc > 0) * log(K);
	return -res;
}

static inline void grad_nll_contrib(double *gbc, double *gbr, double *gac, double *gar, /* Gradients to update */
			double *bc, double *br, double *ac, double *ar, /* Parameters */
                        uint8_t yc, double *xc, size_t dc, /* Data Ca. */
                        uint8_t yr, double *xr, size_t dr) { /* Data Reca. */
        if (!(bc && br && ac && ar && xc && xr && gbc && gbr && gac && gar)) return;
	double fc, pc = *ac;
	double fr, pr = *ar;
	double res = 0.0;
	double K;
	size_t j;
	for (j = 0; j < dc; j++) pc += xc[j] * bc[j];
	for (j = 0; j < dr; j++) pr += xr[j] * br[j];
	pc = logis(pc);
	pr = logis(pr);
	K = 1.0 - pc;
	K *= 1.0 - pr;
	K = 1.0 - K;
	fc = (double) (yc > 0) / pc - (double) (yc == 0 && yr > 0) / (1.0 - pc) - (double) (yr > 0 || yc > 0) * (1.0 - pr) / K;
	fr = (double) (yr > 0) / pr - (double) (yr == 0 && yc > 0) / (1.0 - pr) - (double) (yr > 0 || yc > 0) * (1.0 - pc) / K;
	fc *= pc * (1.0 - pc);
	fr *= pr * (1.0 - pr);
	for (j = 0; j < dc; j++) gbc[j] -= xc[j] * fc;
	for (j = 0; j < dr; j++) gbr[j] -= xr[j] * fr;
	*gac -= fc;
	*gar -= fr;
}

typedef struct alho_data {
	uint8_t *yc; /* Pointer to a vector of binary values */
	double *xc; /* Pointer to a matrix in row-major format */
	uint8_t *yr; /* Pointer to a vector of binary values */
	double *xr; /* Pointer to a matrix in row-major format */
	size_t n; /* Number of observations */
	size_t dc; /* Number of ca. */
	size_t dr; /* Number of reca. */
} alho_data;

static void alho_grad(double *grd, double *param, int *p, void *info) {
	if (!(grd && param && p && info)) return;
	memset(grd, 0, *p * sizeof(double));
	alho_data *dta = (alho_data *) info;
	if ((size_t) *p != (dta->dc + dta->dr + 2)) return;
	for (size_t i = 0; i < dta->n; i++) {
		grad_nll_contrib(&grd[1], &grd[2 + dta->dc], grd, &grd[1 + dta->dc], \
				 &param[1], &param[2 + dta->dc], param, &param[1 + dta->dc], \
	                         dta->yc[i], &dta->xc[i], dta->dc, \
        	                 dta->yr[i], &dta->xr[i], dta->dr);
	}
}

double * alho_fit(int *yc, int *yr, double *xc, double *xr, int *dxc, int *dxr, int *maxit, bool cmc, bool cmr) {
	if (!(yc && yr && xc && xr && dxc && dxr && maxit)) return NULL;
	if (!(*dxc == *dxr && *maxit > 0)) return NULL;
	if (!(dxc[0] > 0 && dxc[1] > 0 && dxr[1] > 0)) return NULL;
	alho_data info;
	size_t i, j;
	int len = 2 + dxc[1] + dxr[1];
	double *param = (double *) calloc(len, sizeof(double));
	info.n = (size_t) *dxc;
	info.yc = (uint8_t *) malloc(info.n * sizeof(uint8_t));
	info.yr = (uint8_t *) malloc(info.n * sizeof(uint8_t));
	info.dc = (size_t) dxc[1];
	info.dr = (size_t) dxr[1];
	info.xc = (double *) malloc(info.n * info.dc * sizeof(double));
	info.xr = (double *) malloc(info.n * info.dr * sizeof(double));
	if (param && info.yc && info.yr && info.xc && info.xr) {
		if (cmc) { /* Transpose covariates */
			#pragma omp parallel for collapse(2)
			for (j = 0; j < info.dc; j++) {
				for (i = 0; i < info.n; i++) {
					info.xc[info.dc * i + j] = xc[info.n * j + i];
				}
			}
		}
		else {
			memcpy(info.xc, xc, sizeof(double) * info.n * info.dc);
		}
		if (cmr) { /* Transpose covariates */
			#pragma omp parallel for collapse(2)
			for (j = 0; j < info.dr; j++) {
				for (i = 0; i < info.n; i++) {
					info.xr[info.dr * i + j] = xr[info.n * j + i];
				}
			}
		}
		else {
			memcpy(info.xr, xr, sizeof(double) * info.n * info.dr);
		}
		/* Process the binary responses */
		#pragma omp parallel for simd
		for (i = 0; i < info.n; i++) {
			info.yc[i] = (uint8_t) yc[i];
			info.yr[i] = (uint8_t) yr[i];
		}
		lion(param, &len, maxit, (void *) &info, alho_grad);
	}
	free(info.yc);
	free(info.yr);
	free(info.xc);
	free(info.xr);	
	return param;
}

#ifdef DEBUG

int main() {
	int i;
	int maxit = 100000;

	int yc[] = {1, 1, 1, 1, 1, 1, 1, 0, 0};
	int yr[] = {0, 0, 0, 1, 1, 1, 1, 1, 1};
	double xc[] = {-0.18, -2.55,  0.69, -0.43,  0.27,  0.49, -0.54,  1.14,  1.71, \
	               -0.51, -3.88, -0.20, -0.84,  0.93, -0.43, -0.04,  0.93,  1.46, \
				    0.49, -3.01, -1.78, -0.84, -0.95, -0.59,  0.32,  0.86,  1.31};
	double xr[] = {-1.22,  0.10,  0.36, -1.47, -0.53,  0.78, -0.08, -0.17,  1.02, \
	                0.42,  0.11,  0.62,  0.89, -0.65, -0.10,  0.39,  1.84, -0.11, \
				   -0.33, -2.15, -0.93,  1.00,  1.19, -1.00,  0.15, -0.59,  0.37, \
				    0.96,  1.74, -0.38,  0.47, -0.87, -0.66,  0.22, -1.31, -0.75};
	int dxc[] = {9, 3};
	int dxr[] = {9, 4};

	double * coef = alho_fit(yc, yr, xc, xr, dxc, dxr, &maxit, true, true);

	if (coef) {
		printf("Alho's coefficients:\n");
		for (i = 0; i < 4; i++) printf("%g ", coef[i]);
		printf("\n");
		for (i = 4; i < 9; i++) printf("%g ", coef[i]);
		printf("\n");
		free(coef);
	}
	return 0;
}

#endif

