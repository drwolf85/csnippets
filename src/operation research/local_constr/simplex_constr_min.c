#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RHO .9
#define EPSILON 1e-5

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
    double *dlt_v;
    double *mom_s;

    grd_v = (double *) malloc(p * sizeof(double));
    dlt_v = (double *) calloc(p, sizeof(double));
    mom_s = (double *) calloc(p, sizeof(double));
    if (mom_s && grd_v && dlt_v) {
		for (t = 0; t < n_iter; t++) {
			grad(grd_v, w, p, info); /* Compute the gradient */
			for (i = 0; i < p; i++) grd_v[i] *= w[i] * (1.0 * w[i]);/* Adjust the gradient to account for the transformation performed at the next line */
			conv_from_simplex(w, w, p); /* Convert vector from Simplex space to an Euclidean space */
			for (i = 0; i < p; i++) { /* Adadelta descent step */
				/* Update second order momentum */
                mom_s[i] *= RHO;
                mom_s[i] += (1.0 - RHO) * grd_v[i] * grd_v[i];
                /* Computing the step */
                grd_v[i] *= sqrt((dlt_v[i] + EPSILON) / (mom_s[i] + EPSILON));
                w[i] -= grd_v[i];
                /* Update the deltas */
                dlt_v[i] *= RHO;
                dlt_v[i] += (1.0 - RHO) * grd_v[i] * grd_v[i];
            }
			conv_to_simplex(w, w, p); /* Transform the unknowns back to the Simplex space*/
		}
	}
    free(mom_s);
    free(grd_v);
    free(dlt_v);
}

