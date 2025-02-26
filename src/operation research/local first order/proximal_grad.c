#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BETA_1 0.9
#define BETA_2 0.999
#define FACTOR_P 0.01
#define LEARNING_RATE 0.001

/* This program is used to minimize `f(x) + g(x)`, where `f` is
   convex differentiable and `g` is convex nondifferentiable */

#define M_EPS 1e-9
#define M_H_EPS 1e-3


static inline void get_grad(double *grd, int n, 
					 double *x, double *v, double lambda,
					 double (*g)(double *, int, void *), 
					 void *info, double step_sz) {
	int i;
	double const il = 1.0 / lambda;
	double const nrm = 1.0 / step_sz;
	for (i = 0; i < n; i++) {
		x[i] += step_sz;
		grd[i] = g(x, n, info);
		x[i] -= step_sz;
		grd[i] -= g(x, n, info);
		grd[i] *= nrm;
		grd[i] += il * (x[i] - v[i]);
	}
}

/* res = arg min_x (g(x) + 0.5 / lambda * l2norm(x-v) ** 2 */
static inline void proximal_operator(double *res, double *v, double *mom_m,
								     int n, double *grd_v, double lambda,
                                     double (*g)(double *, int, void *),
                                     void *info, int maxit, double step_sz) {
	int i, t;
	double sgn;
	for (t = 0; t < maxit; t++) {
		/* Update the gradient */
		memset(mom_m, 0, sizeof(double) * n);
		get_grad(grd_v, n, res, v, lambda, g, info, step_sz);
		for (i = 0; i < n; i++) {
			/* Lion update of custom momentum */
			sgn = BETA_1 * mom_m[i] + (1.0 - BETA_1) * grd_v[i];
			/* Update the momentum */
			mom_m[i] *= BETA_2;
			mom_m[i] +=  (1.0 - BETA_2) * grd_v[i];
			/* Lion update */
			sgn = (double) (sgn > 0.0) - (double) (sgn < 0.0);
			/* Computing the step */
			grd_v[i] = sgn + FACTOR_P * res[i];
			grd_v[i] *= LEARNING_RATE;
			res[i] -= grd_v[i];
		}
	}
}

/* min f(x) + g(x) */
void proximal_grad(double *x, int n, double step_sz, double lambda, double alpha,
                   void (*f_grd)(double *, double *, int, void *),
                   double (*g)(double *, int, void *), void *info, int maxit, int maxop) {
	int i, k;
	double test = 2.0 * fabs(M_EPS) + 1.0;
	double *v = (double *) calloc(n, sizeof(double));
	double *grd = (double *) malloc(n * sizeof(double));
	double *grd2 = (double *) malloc(n * sizeof(double));
	double *mom2 = (double *) malloc(n * sizeof(double));
	if (v && grd && grd2 && mom2 && x && f_grd && g && alpha > 0.0 && lambda > 0.0) {
		for (k = 0; k < maxit && test > M_EPS; k++) {
			f_grd(grd, x, n, info);
			for (i = 0; i < n; i++)
				v[i] = x[i] - alpha * grd[i];
			proximal_operator(x, v, mom2, n, grd2, lambda, g, info, maxop, step_sz);
			test = 0.0;
			for (i = 0; i < n; i++) {
				v[i] = x[i] - (v[i] + alpha * grd[i]);
				test += v[i];
			}
			test = fabs(sqrt(fabs(test)));
		}
	}
	if (v) free(v);
	if (grd) free(grd);
	if (grd2) free(grd2);
	if (mom2) free(mom2);
}

#ifdef DEBUG
typedef struct info_t {
	double *y;
	double *x; 
	int n;
	int p;
} info_t;

void my_f_grad(double *grad_f, double *par, int p, void *info) {
	int i, j;
	double tmp;
	info_t nf = *(info_t *) info;
	memset(grad_f, 0, sizeof(double) * p);
	for (i = 0; i < nf.n; i++) {
		tmp = par[0];
		for (j = 0; j < nf.p; j++) tmp += par[j + 1] * nf.x[j + nf.p * i];
		tmp = 1.0 / (1.0 + exp(-tmp));
		tmp = nf.y[i] - tmp;
		grad_f[0] += tmp;
		for (j = 0; j < nf.p; j++) grad_f[j] += tmp * nf.x[j + nf.p * i];
	}
}

double my_g(double *par, int p, void *info) {
	double res = 0.0;
	int i;
	for (i = 1; i < p; i++) {
		res += fabs(par[i]);
	}
	return res;
}

int main() {
	int const N = 20;
	int const P = 2;
	double x[] = {0.57, -0.15, 1.4, -0.5, 0.52, -0.68, -1.79, -0.13, 0.06, 0.67, 0.1, 0.92, 0.04, -0.46, 0.81, -0.84, -2.14, -0.55, 0.95, 0.28, 0.31, 0.43, 1.63, -0.12, 0.94, 1.28, -0.88, -0.83, -2.11, -1.05, -0.06, -0.19, -0.01, 1.07, -1.38, -1.8, -0.41, -1.78, 1.28, -0.88};
	double y[] = {0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
	double par[3] = {0.0, -3.2, 3.2};
	info_t info = {y, x, N, P};
	proximal_grad(par, 3, 1e-3, 100.0, 1e-3, my_f_grad, my_g, (void *) &info, 1000L, 5L);
	printf("Final param: ");
	for (int i = 0; i < 3; i++) printf("%g ", par[i]);
	printf("\n");
	return 0;
}
#endif
