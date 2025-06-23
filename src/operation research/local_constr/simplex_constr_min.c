#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.01
#define BETA_1 0.9
#define BETA_2 0.999
#define FACTOR_P 0.01

typedef struct simplex_vec {
	double v;
	unsigned i;
} vex;

static int cmp_rev_vex(void const *aa, void const *bb) {
	vex a = *(vex *) aa;
	vex b = *(vex *) bb;
	return 2 * (b.v > a.v) - 1;
}

static inline void conv_from_simplex(double *dst, double *src, unsigned len) {
	unsigned i;
	vex *svx = calloc(len, sizeof(vex));
	if (svx && dst && src) {
		for (i = 0; i < len; i++) {
			svx[i].v = src[i];
			svx[i].i = i;
		}
		qsort(svx, len, sizeof(vex), cmp_rev_vex);
		for (i = 0; i < len; i++) {
			dst[svx[i].i] = log(svx[i].v / svx[0].v);
		}
	}
	free(svx);
}

static inline void conv_to_simplex(double *dst, double *src, unsigned len) {
	unsigned i;
	double sm = 0.0;
	if (dst && src) {
		for (i = 0; i < len; i++) {
			dst[i] = exp(src[i]);
			sm += dst[i];
		}
		sm = 1.0 / sm;
		for (i = 0; i < len; i++) dst[i] *= sm;
	}
}

/*extern void adj_grad_by_simplex_transf(double *g, double *xw, unsigned p) {
	unsigned i;
	double tmp, sm = 0.0, *v;
	v = (double *) calloc(p, sizeof(double));
	if (g && xw && v) {
		for (i = 0; i < p; i++) {
			v[i] = exp(xw[i]);
			sm += v[i];
		}
		sm = 1.0 / sm;
		for (i = 0; i < p; i++) {
			tmp = v[i] * sm;
			g[i] *= tmp * (1.0 - tmp) ;
		}
	}
	free(v);
}*/

extern void min_within_simplex(double *w, unsigned p, unsigned n_iter, void *info, 
						       void (*grad)(double *, double *, unsigned, void *)) {
	unsigned i, t;
    double *grd_v;
    double *mom_m;
    double sgn;

    grd_v = (double *) malloc(p * sizeof(double));
    mom_m = (double *) calloc(p, sizeof(double));
    if (mom_m && grd_v) {
		for (t = 0; t < n_iter; t++) {
			grad(grd_v, w, p, info); /* Compute the gradient */
			for (i = 0; i < p; i++) grd_v[i] *= w[i] * (1.0 * w[i]);/* Adjust the gradient to account for the transformation performed at the next line */
			conv_from_simplex(w, w, p); /* Convert vector from Simplex space to an Euclidean space */
      for (i = 0; i < p; i++) { /* Lion descent step */
        /* Lion update of custom momentum */
        sgn = BETA_1 * mom_m[i] + (1.0 - BETA_1) * grd_v[i];
        /* Update the momentum */
        mom_m[i] *= BETA_2;
        mom_m[i] +=  (1.0 - BETA_2) * grd_v[i];
        /* Lion update */
        sgn = (double) (sgn > 0.0) - (double) (sgn < 0.0);
        /* Computing the step */
        grd_v[i] = sgn + FACTOR_P * w[i];
        grd_v[i] *= LEARNING_RATE;
				w[i] -= grd_v[i];
      } /* End of Lion step */
			conv_to_simplex(w, w, p); /* Transform the unknowns back to the Simplex space*/
		}
	}
  free(mom_m);
	free(grd_v);
}

#ifdef DEBUG

double obj(double *w, double *mu, double *sg, unsigned n) {
	unsigned i, j;
	double res = 0.0;
	for (i = 0; i < n; i++) {
		res -= mu[i] * w[i];
		for (j = 0; j < n; j++) res += 0.5 * w[i] * w[j] * sg[i + j * n];
	}
	return res;
}

struct test_data {
	double *mu;
	double *sg;
};

void obj_grad(double *g, double *w, unsigned n, void *info) {
	unsigned i, j;
	double res = 0.0;
	struct test_data *dta = (struct test_data *) info;
	for (i = 0; i < n; i++) {
		g[i] = -dta->mu[i];
		for (j = 0; j < n; j++) g[i] += w[j] * dta->sg[i + j * n];
	}
}

#include <time.h>

int main() {
	double mu[] = {82.27257876, 0.05543794, -0.57487430, 20.24679474, -0.75812793};
	double sg[] = {5.762207, 4.853735, 3.840732, 5.284037, 4.217663, 8.766472, 10.40729, 6.757895, 8.116401, 8.442137, 5.1568, 5.023761, 7.736688, 4.872962, 4.075185, 8.070748, 6.863769, 5.543381, 8.801097, 6.213287, 7.564125, 8.382827, 5.443372, 7.295588, 10.33417};
	double sm = 0.0, w[5];
	struct test_data dta = {mu: mu, sg: sg};
	unsigned i;
	srand(time(NULL));
	for (i = 0; i < 5; i++) {
		w[i] = -log((0.5 + (double) rand()) / (1.0 + (double) RAND_MAX));
		sm += w[i];
	}
	sm = 1.0 / sm;
	for (i = 0; i < 5; i++) w[i] *= sm;
	printf("Initial values: "); for (i = 0; i < 5; i++) printf("%e ", w[i]);
	printf("Objective: %f\n", obj(w, dta.mu, dta.sg, 5));
	printf("\n");
	min_within_simplex(w, 5, 30000, &dta, obj_grad);
	printf("Final values: "); for (i = 0; i < 5; i++) printf("%e ", w[i]); printf("\n");
	printf("Objective: %f\n", obj(w, dta.mu, dta.sg, 5));
	return 0;
}

#endif
