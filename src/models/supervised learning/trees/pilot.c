/**
 * @brief Fast Linear Model Trees by PIecewise Linear Organic Tree (PILOT)
 *        based on the work of Raymaekers, Rousseeuw, Verdonck, and Yao 
 *        published on July 8, 2024.
 */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* Model types */
typedef enum Model_type { CON, /* CONstant. The recursion stops after fitting this model. */
  LIN, /* LINear regression (the standard one). */
  PCON, /* Piecewise CONstant fit, as in CART. */
  BLIN, /* Broken LINear: a continuous curve based on two linear models. */
  PLIN } model_t; /* Piecewise LINear fit that can be discontinuous. */

typedef struct parameters {
    double con_par; /* Constant predictor (or model intercept) */
    double lin_par; /* Linear coefficient */
} param_t;

typedef struct leaf_model {
    double best_pivot; /* Best pivot value (a.k.a. the splitting value) */
    model_t best_model; /* Best model type */
    double best_bic; /* Best BIC */
    size_t best_x; /* Best predictor */
    param_t lm_l; /* Parameters for the left side */
    param_t lm_r; /* Parameters for the right side */
    double range[2]; /* Range of the best predictor */
} leaf_model;

typedef struct node {
    leaf_model mod;
    size_t depth;
    struct node *l, *r;
} node;

typedef struct full_tree {
	node *T;
	double *xC;
	size_t p;
	double B;
	double C;
} tree_t;

/**
 * @brief Function to truncate a value according to its upper and lower bounds
 * 
 * @param x Generic double precision value
 * @param lw The lower truncation bound 
 * @param up The upper truncation bound 
 * @return double 
 */
double clamp(double x, double lw, double up) {
    double res = x;
    if (x > up) return up;
    if (x < lw) return lw;
    return res;
}

/**
 * @brief Range function to compute min and max of a vector
 *
 * @param r Pointer to an array of length two
 * @param x Pointer to a vector of values
 * @param n Number of values in `x`
 */
static inline void range(double *r, double *x, size_t n) {
	double mx, mn;
	mx = mn = x[0];
	for (size_t i = 1; i < n; i++) {
		 /* mx += (double) (mx < x[i]) * (x[i] - mx);
		mn += (double) (x[i] < mn) * (x[i] - mn); */
		if (mx < x[i]) mx = x[i];
		if (mn > x[i]) mn = x[i];
	}
	r[0] = mn;
	r[1] = mx;
}


/**
 * @brief Which min in a vector
 *
 * @param x Pointer to a vector of values
 * @param n Number of values in `x`
 */
static inline size_t which_min(double *x, size_t n) {
	double mn;
	size_t wh = 0;
	mn = x[wh];
	for (size_t i = 1; i < n; i++) {
		wh += (size_t) (x[i] < mn) * (i - wh);
		mn += (double) (x[i] < mn) * (x[i] - mn);
	}
	return wh;
}


/**
 * @brief Function to center the values in a vector
 *
 * @param x Pointer to a vector of values
 * @param n Number of values in `x`
 * @param B Pointer to a value where to store the range
 * @param C Pointer to a value where to store the original central value
 * @return Pointer to double
 */
double * center(double *x, size_t n, double *B, double *C) {
    double *y0 = (double *) calloc(n, sizeof(double));
    double rng[2] = {0};
    size_t i;
    if (x && y0) {
    	range(rng, x, n);
        *B = 0.5 * (rng[1] - rng[0]);
        *C = 0.5 * (rng[1] + rng[0]);
        for (i = 0; i < n; i++) y0[i] = x[i] - *C;
    }
    return y0;
}

/**
 * @brief Function to generate a sequence of indices
 *
 * @param n Number of integer numbers to generate as a vector of indices
 * @return Pointer to size_t
 */
size_t * seq_len(size_t n) {
    size_t * ivec = (size_t *) calloc(n, sizeof(size_t));
    if (ivec) for (size_t i = 0; i < n; i++) ivec[i] = i;
    return ivec;
}



/**
 * @brief "Negative copy"
 * @param dest Destination vector (double)
 * @param src Source vector (double)
 * @param n length of the two vectors above (size_t)
 * @param sgn Character determining how to treat negative numbers
 */
static inline void negcpy(double *dest, double *src, size_t n, char sgn) {
    size_t i;
    if (sgn) { /* Part to execute at the end when `sgn != 0` */
        for (i = 0; i < n; i++) {
            dest[i] = -src[i];
            if (src[i] > 0.0)
                dest[i] = 1.0 / dest[i];
        }
    }
    else { /* Part to execute at the beginning when `sgn == 0` */
        for (i = 0; i < n; i++) {
            dest[i] = -src[i];
            if (src[i] < 0.0)
                dest[i] = 1.0 / dest[i];
        }
    }
}

/**
 * @brief Sorting a vector of real number (from
 *        smallest to largest) with indexes
 * @param x Vector of real numbers (double)
 * @param idx Vector of indices (integer numbers, size_t)
 * @param rnk Vector of ranks (integer numbers, size_t)
 * @param n length of the vector above (size_t)
 */
static void sort_reals_widnr(double *x, size_t *idx, size_t *rnk, size_t n) {
    double *s, *y;
    uint8_t c;
    uint64_t v;
    size_t i, h[256], ch[256], *sid;
    sid = (size_t *) malloc(n * sizeof(size_t));
    s = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));

    if (s && y && sid && x && idx && rnk) {
    	for (i = 0; i < n; i++) idx[i] = i;
        negcpy(s, x, n, 0);
        for (c = 0; c < sizeof(double); c++) {
            memset(h,  0, 256 * sizeof(size_t));
            memset(ch, 0, 256 * sizeof(size_t));
            /* build histogram counts */
            for (i = 0; i < n; i++) {
                v = *(uint64_t *) &s[i];
                v >>= 8 * c;
                v &= 255;
                h[v]++;
            }
            /* adjust starting positions */
            for (i = 1; i < 256; i++) {
                ch[i] = ch[i-1] + h[i-1];
            }
            /* sort values */
            for (i = 0; i < n; i++) {
                v = *(uint64_t *) &s[i];
                v >>= 8 * c;
                v &= 255;
                y[ch[v]] = s[i];
                sid[ch[v]] = idx[i];
                ch[v]++;
            }
            /* Copy to sorted vectors */
            memcpy(s, y, n * sizeof(double));
            memcpy(idx, sid, n * sizeof(size_t));
        }
        for (i = 0; i < n; i++) rnk[idx[i]] = i;
        /* negcpy(x, s, n, 1); */ /** NOTE: x does not need to be sorted at the end of this function */
    }
    free(sid);
    free(s);
    free(y);
}

/**
 * @brief Comparison function to quick-sort the values in `ivec`
 * 
 * @param aa Pointer to the first item to compare
 * @param bb Pointer to the second item to compare
 * 
 * @return int
 */
static int cmp_ivec(void const *aa, void const *bb) {
	size_t a = *(size_t *) aa;
	size_t b = *(size_t *) bb;
	return (int) (a > b) * 2 - 1;
}

/** 
 * @brief Function to select a model with the lowest BIC value
 * 
 * @param mod Pointer to a `leaf_model` structure
 * @param ivec Pointer to a vector of indices to select the cases in the node
 * @param rnk Pointer to a vector of the ranks of variable `x`
 * @param idx Pointer to a vector of the sorted indices of variable `x`
 * @param n Length of `ivec`
 * @param x Pointer to a variable `x`
 * @param y Pointer to a vector of residuals
 * @param n_leaf Pointer to the minimal number of cases in a terminal leaf
 * @param j Index of the variable to process
 */
static inline void BIC_select(leaf_model *mod, size_t *ivec, size_t *idx, size_t *rnk, size_t n, double *x, double *y, size_t const *n_leaf, size_t j) {
	size_t i = 0;
	double const ninv = 1.0 / (double) n;
	double inl, inmnl;
	double sx[2] = {0};
	double sy[2] = {0};
	double sxx[2] = {0};
	double sxy[2] = {0};
	double syy[2] = {0};
	size_t const n_tests = n  - *n_leaf * 2; 
	double par[4] = {0};
	double bic;
	size_t *R = (size_t *) malloc(n * sizeof(size_t));

	if (R) {
		for (i = 0; i < n; i++) R[i] = rnk[ivec[i]];
		qsort(R, n, sizeof(size_t), cmp_ivec);
		for (i = 0; i < n; i++) R[i] = idx[R[i]];
		/* Compute sufficient statistics */
		for (i = 0; i < *n_leaf; i++) {
			sx[0]  += x[R[i]];
			sy[0]  += y[R[i]];
			sxx[0] += x[R[i]] * x[R[i]];
			sxy[0] += y[R[i]] * x[R[i]];
			syy[0] += y[R[i]] * y[R[i]];
		}
		for (; i < n; i++) {
			sx[1]  += x[R[i]];
			sy[1]  += y[R[i]];
			sxx[1] += x[R[i]] * x[R[i]];
			sxy[1] += y[R[i]] * x[R[i]];
			syy[1] += y[R[i]] * y[R[i]];
		}
		/* Compute parameter estimates and BIC for constant */
		par[0] = ninv * (sy[0] + sy[1]);
		bic = (double) n * log((syy[0] + syy[1]) * ninv - par[0] * par[0]) + log((double) n); /* BIC_CON */
		if (bic < mod->best_bic) {
			mod->best_model = CON;
			mod->best_x = j;
			mod->best_pivot = INFINITY;
			mod->lm_l.con_par = par[0];
			mod->lm_l.lin_par = 0.0;
			mod->lm_r.con_par = par[0];
			mod->lm_r.lin_par = 0.0;
			mod->best_bic = bic;
		}
		// /* Compute parameter estimates and BIC for linear */
		par[2] = ninv * (sy[0] + sy[1]);
		par[0] = sx[0] + sx[1];
		par[1] = (sxy[0] + sxy[1] - par[0] * par[2]) / (sxx[0] + sxx[1] - par[0] * par[0] * ninv);
		par[0] = par[2] - par[1] * par[0] * ninv;
		bic = syy[0] + syy[1] + par[0] * par[0] * (double) n;
		bic += (sxx[0] + sxx[1]) * par[1] * par[1];
		bic -= 2.0 * (par[0] * (sy[0] + sy[1]) + par[1] * (sxy[0] + sxy[1]) - par[0] * par[1] * (sx[0] + sx[1]));
		bic = (double) n * log(bic * ninv) + 2.0 * log((double) n); /* BIC_LIN */
		if (bic < mod->best_bic) {
			mod->best_model = LIN;
			mod->best_x = j;
			mod->best_pivot = INFINITY;
			mod->lm_l.con_par = par[0];
			mod->lm_l.lin_par = par[1];
			mod->lm_r.con_par = par[0];
			mod->lm_r.lin_par = par[1];
			mod->best_bic = bic;
		}
		inl = 1.0 / (double) *n_leaf;
		inmnl = 1.0 / (double) (n - *n_leaf);
		/* Compute parameter estimates and BIC for piecewise constant */
		par[0] = sy[0] * inl;
		par[2] = sy[1] * inmnl;
		bic = syy[0] * inl - par[0] * par[0];
		bic += syy[1] * inmnl - par[2] * par[2];
		bic = (double) n * log(bic) + 5.0 * log((double) n); /* BIC_PCON */
		if (bic < mod->best_bic) {
			mod->best_model = PCON;
			mod->best_x = j;
			mod->best_pivot = x[*n_leaf];
			mod->lm_l.con_par = par[0];
			mod->lm_l.lin_par = 0.0;
			mod->lm_r.con_par = par[2];
			mod->lm_r.lin_par = 0.0;
			mod->best_bic = bic;
		}
		// /* Compute parameter estimates and BIC for broken linear */
		par[1] = (sxy[0] - sx[0] * sy[0] * inl) / (sxx[0] - sx[0] * sx[0] * inl);
		par[2] = (sxy[1] - sx[1] * sy[1] * inmnl) / (sxx[1] - sx[1] * sx[1] * inmnl);
		par[0] = ninv * (sy[0] + sy[1]) - par[1] * sx[0] * inl - par[2] * sx[1] * inmnl;
		bic  = par[0] * par[1] * sx[0];
		bic += par[0] * par[2] * sx[1];
		bic -= par[1] * sxy[0];
		bic -= par[2] * sxy[1];
		bic -= par[0] * (sy[0] + sy[1]);
		bic *= 2.0;
		bic += syy[0] + syy[1] + par[0] * par[0] * (double) n;
		bic += sxx[0] * par[1] * par[1];
		bic += sxx[1] * par[2] * par[2];
		bic = (double) n * log(bic * ninv) + 5.0 * log((double) n); /* BIC_BLIN */
		if (bic < mod->best_bic) {
			mod->best_model = BLIN;
			mod->best_x = j;
			mod->best_pivot = x[*n_leaf];
			mod->lm_l.con_par = par[0];
			mod->lm_l.lin_par = par[1];
			mod->lm_r.con_par = par[0];
			mod->lm_r.lin_par = par[2];
			mod->best_bic = bic;
		}
		/* Compute parameter estimates and BIC for piecewise linear */
		par[3] = par[2];
		par[0] = (sy[0] - par[1] * sx[0]) * inl;
		par[2] = (sy[1] - par[3] * sx[1]) * inmnl;
		bic  = par[0] * par[1] * sx[0];
		bic += par[2] * par[3] * sx[1];
		bic -= par[1] * sxy[0];
		bic -= par[3] * sxy[1];
		bic -= par[0] * sy[0];
		bic -= par[2] * sy[1];
		bic *= 2.0;
		bic += syy[0] + syy[1];
		bic += par[0] * par[0] * (double) (*n_leaf);
		bic += par[2] * par[2] * (double) (n - *n_leaf);
		bic += sxx[0] * par[1] * par[1];
		bic += sxx[1] * par[3] * par[3];
		bic = (double) n * log(bic * ninv) + 7.0 * log((double) n); /* BIC_PLIN */
		if (bic < mod->best_bic) {
			mod->best_model = BLIN;
			mod->best_x = j;
			mod->best_pivot = x[*n_leaf];
			mod->lm_l.con_par = par[0];
			mod->lm_l.lin_par = par[1];
			mod->lm_r.con_par = par[2];
			mod->lm_r.lin_par = par[3];
			mod->best_bic = bic;
		}

		for (i = 0; i < n_tests; i++) {
			/* Update sufficient statistics */
			sx[0]  += x[R[*n_leaf + i]];
			sy[0]  += y[R[*n_leaf + i]];
			sxx[0] += x[R[*n_leaf + i]] * x[R[*n_leaf + i]];
			sxy[0] += y[R[*n_leaf + i]] * x[R[*n_leaf + i]];
			syy[0] += y[R[*n_leaf + i]] * y[R[*n_leaf + i]];
			sx[1]  -= x[R[*n_leaf + i]];
			sy[1]  -= y[R[*n_leaf + i]];
			sxx[1] -= x[R[*n_leaf + i]] * x[R[*n_leaf + i]];
			sxy[1] -= y[R[*n_leaf + i]] * x[R[*n_leaf + i]];
			syy[1] -= y[R[*n_leaf + i]] * y[R[*n_leaf + i]];
			inl = 1.0 / (double) (*n_leaf + i + 1);
			inmnl = 1.0 / (double) (n - *n_leaf - i - 1);
			/* Compute parameter estimates and BIC for piecewise constant */
			par[0] = sy[0] * inl;
			par[2] = sy[1] * inmnl;
			bic = syy[0] * inl - par[0] * par[0];
			bic += syy[1] * inmnl - par[2] * par[2];
			bic = (double) n * log(bic) + 5.0 * log((double) n); /* BIC_PCON */
			if (bic < mod->best_bic) {
				mod->best_model = PCON;
				mod->best_x = j;
				mod->best_pivot = x[*n_leaf + i + 1];
				mod->lm_l.con_par = par[0];
				mod->lm_r.con_par = par[2];
				mod->best_bic = bic;
			}
			// /* Compute parameter estimates and BIC for broken linear */
			par[1] = (sxy[0] - sx[0] * sy[0] * inl) / (sxx[0] - sx[0] * sx[0] * inl);
			par[2] = (sxy[1] - sx[1] * sy[1] * inmnl) / (sxx[1] - sx[1] * sx[1] * inmnl);
			par[0] = ninv * (sy[0] + sy[1]) - par[1] * sx[0] * inl - par[2] * sx[1] * inmnl;
			bic  = par[0] * par[1] * sx[0];
			bic += par[0] * par[2] * sx[1];
			bic -= par[1] * sxy[0];
			bic -= par[2] * sxy[1];
			bic -= par[0] * (sy[0] + sy[1]);
			bic *= 2.0;
			bic += syy[0] + syy[1] + par[0] * par[0] * (double) n;
			bic += sxx[0] * par[1] * par[1];
			bic += sxx[1] * par[2] * par[2];
			bic = (double) n * log(bic * ninv) + 5.0 * log((double) n); /* BIC_BLIN */
			if (bic < mod->best_bic) {
				mod->best_model = BLIN;
				mod->best_x = j;
				mod->best_pivot = x[*n_leaf + i + 1];
				mod->lm_l.con_par = par[0];
				mod->lm_l.lin_par = par[1];
				mod->lm_r.con_par = par[0];
				mod->lm_r.lin_par = par[2];
				mod->best_bic = bic;
			}
			/* Compute parameter estimates and BIC for piecewise linear */
			par[3] = par[2];
			par[0] = (sy[0] - par[1] * sx[0]) * inl;
			par[2] = (sy[1] - par[3] * sx[1]) * inmnl;
			bic  = par[0] * par[1] * sx[0];
			bic += par[2] * par[3] * sx[1];
			bic -= par[1] * sxy[0];
			bic -= par[3] * sxy[1];
			bic -= par[0] * sy[0];
			bic -= par[2] * sy[1];
			bic *= 2.0;
			bic += syy[0] + syy[1];
			bic += par[0] * par[0] * (double) (*n_leaf + i + 1);
			bic += par[2] * par[2] * (double) (n - *n_leaf - i - 1);
			bic += sxx[0] * par[1] * par[1];
			bic += sxx[1] * par[3] * par[3];
			bic = (double) n * log(bic * ninv) + 7.0 * log((double) n); /* BIC_PLIN */
			if (bic < mod->best_bic) {
				mod->best_model = BLIN;
				mod->best_x = j;
				mod->best_pivot = x[*n_leaf + i + 1];
				mod->lm_l.con_par = par[0];
				mod->lm_l.lin_par = par[1];
				mod->lm_r.con_par = par[2];
				mod->lm_r.lin_par = par[3];
				mod->best_bic = bic;
			}
		}
		free(R);
	}
}

/**
 * @brief Function to select a model by screening all variables in `X`
 * 
 * @param err Pointer to a vector of residuals
 * @param X Pointer to a matrix of covariates
 * @param n Number of cases to process (i.e., length of `ivec`)
 * @param p Pointer to the number of variables in `X`
 * @param idx Pointer to a matrix of sorted indices of the values in `X`
 * @param rnk Pointer to a matrix of ranks of each variable in `X`
 * @param ivec Pointer to the cases to process for the newly allocated node 
 * @param n_leaf Pointer to the minimal number of cases in a terminal leaf
 * @param N Pointer to the total number of cases in `X`
 * 
 * @return Pointer to a newly allocated tree node
 */
node * model_select(double *err, double *X, size_t const n, size_t const *p, size_t *idx, size_t *rnk, size_t *ivec, 
		    size_t const *n_leaf, size_t const *N) {
	node *best = (node *) calloc(1, sizeof(node));
	size_t j, i;
	if (best) {
		best->mod.best_bic = INFINITY;
		for(j = 0; j < *p; j++) {
			BIC_select(&best->mod, ivec, &idx[*N * j], &rnk[*N * j], n, &X[*N * j], err, n_leaf, j);
		}
		j = *N * best->mod.best_x;
		best->mod.range[0] = best->mod.range[1] = X[j + ivec[0]];
		for (i = 0; i < n; i++) {
			best->mod.range[0] += (double) (best->mod.range[0] > X[j + ivec[i]]) * (X[j + ivec[i]] - best->mod.range[0]);
			best->mod.range[1] += (double) (best->mod.range[1] < X[j + ivec[i]]) * (X[j + ivec[i]] - best->mod.range[1]);
		}
	}
	return best;
}
/**
 * @brief Function to separate cases from the left and the right branches
 * 
 * @param ivec Pointer to a vector of indices
 * @param n Length of the vector of indices
 * @param T Pointer to a tree node
 * @param X Pointer to the matrix of covariates
 * @param N Total number of rows in `X`
 * 
 * @return size_t (Location of the first index store for the right branch)
 */
size_t fix_ivec(size_t *ivec, size_t n, node *T, double *X, size_t const *N) {
	size_t i, elk, ell, end = n - 1;
	double xl, xk;
	bool unswapped;
	i = 0;
	while(i < end) {
		unswapped = true;
		xl = X[*N * T->mod.best_x + ivec[i]];
		if (xl >= T->mod.best_pivot) {
			do {
				xk = X[*N * T->mod.best_x + ivec[end]];
				if (xk < T->mod.best_pivot) {
					ivec[i] ^= ivec[end];
					ivec[end] ^= ivec[i];
					ivec[i] ^= ivec[end];
					unswapped = false;
				}
				--end;
			} while(unswapped && end > i);
		}
		i++;
	}
	end = 0;
	for (i = 0; i < n; i++) {
		xl = X[*N * T->mod.best_x + ivec[i]];
		end += (size_t) (xl < T->mod.best_pivot);
	}
	return end;
}

/**
 * @brief Function to build a tree
 * 
 * @param raw_pred Pointer to a vector where to store raw predictions
 * @param y Pointer to the vector of response values
 * @param err Pointer to a vector of residuals
 * @param X Pointer to a matrix of covariates
 * @param n Number of cases to process when building the node
 * @param p Pointer to the number of variables
 * @param K Number of the current depth level
 * @param idx Pointer to a matrix of sorted indices of the values in `X`
 * @param rnk Pointer to a matrix of ranks of the variables in `X`
 * @param ivec Pointer to a vector of indices related to extract proper cases for the training
 * @param k_max Pointer to the maximal depth allowed
 * @param n_fit Pointer to the minimal number of cases used to fit a model
 * @param n_leaf Pointer to the minimal number of cases in terminal leaves
 * @param B Pointer to a value acting as a bound
 * @param N Pointer to the total number of cases in `X` and `rnk`
 * @param clamping Pointer to a boolean value. If this values is true, the clamp function is activated.
 * @return A pointer to a node/branch structure
 */
node * tree_build(double *raw_pred, double *y, double *err, double *X, 
                  size_t const n, size_t const *p, size_t K, size_t *idx, size_t *rnk, size_t *ivec, 
		  size_t const *k_max, size_t const *n_fit, size_t const *n_leaf, 
		  double const *B, size_t const *N, bool const *clamping) {
	node *T = NULL;
	double x;
	size_t i, ell;
	if (K < *k_max && n >= *n_fit && *n_leaf > 2 && *n_fit > *n_leaf * 2) {
		T = model_select(err, X, n, p, idx, rnk, ivec, n_leaf, N);
		T->depth = K;
		if (T->mod.best_model == CON) {
			#ifdef DEBUG
			for (i = 0; i < n; i++) { /* Predictions of the constant model */
				ell = ivec[i];
				raw_pred[ell] = T->mod.lm_l.con_par;
				/* Residual update */
				if (*clamping) {
					err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], *B * (-1.5), *B * 1.5);
				}
				else {
					err[ell] -= raw_pred[ell];
				}
			}
			#endif
		}
		else if (T->mod.best_model == LIN) {
			for (i = 0; i < n; i++) { /* Predictions of the linear model */
				ell = ivec[i];
				x = X[*N * T->mod.best_x + ell];
				raw_pred[ell] = T->mod.lm_l.con_par + T->mod.lm_l.lin_par * x;
				/* Residual update */
				if (*clamping) {
					err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], *B * (-1.5), *B * 1.5);
				}
				else {
					err[ell] -= raw_pred[ell];
				}
			}
			T->l = tree_build(raw_pred, y, err, X, n, p, K, idx, rnk, ivec, k_max, n_fit, n_leaf, B, N, clamping);
		}
		else { /* Processing Threshold models for growing branches*/
			K++;
			for (i = 0; i < n; i++) {
				ell = ivec[i];
				x = X[*N * T->mod.best_x + ell];
				raw_pred[ell]  = (double) (x <  T->mod.best_pivot) * (T->mod.lm_l.con_par + T->mod.lm_l.lin_par * x);
				raw_pred[ell] += (double) (x >= T->mod.best_pivot) * (T->mod.lm_r.con_par + T->mod.lm_r.lin_par * x);
				/* Residual update */
				if (*clamping) {
					err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], *B * (-1.5), *B * 1.5);
				}
				else {
					err[ell] -= raw_pred[ell];
				}
			}
			/* Fix the vector of indices to separate left and right nodes */
			ell = fix_ivec(ivec, n, T, X, N);
			qsort(ivec, ell, sizeof(size_t), cmp_ivec);
			qsort(&ivec[ell], n - ell, sizeof(size_t), cmp_ivec);
			T->l = tree_build(raw_pred, y, err, X, ell, p, K, idx, rnk, ivec, k_max, n_fit, n_leaf, B, N, clamping);
			T->r = tree_build(raw_pred, y, err, X, n - ell, p, K, idx, rnk, &ivec[ell], k_max, n_fit, n_leaf, B, N, clamping);
		}
	}
	return T;
}

/**
 * @brief Pilot training function 
 * 
 * @param y Pointer to a vector of response values
 * @param X Pointer to a matrix of covariates (stored in a column-major format)
 * @param dimX Pointer to the number of rows and columns of `X`
 * @param Kmax Pointer to the number of maximum depth of the tree
 * @param nFit Pointer to the minimal number of cases required to fit a model
 * @param nLeaf Pointer to the minimal number of cases in a leaf
 * @param centering Boolean values. If `false`, the values in `x` are already centered. If `true`, the values of `x` are then centered.
 * @param clamping Boolean value. If true activate clamping of predicted values
 * @return A Pointer to a tree structure with the fin
 */
tree_t * pilot(double *y, double const *X, int *dimX, int *Kmax, int *nFit, int *nLeaf, bool const centering, bool const clamping) {
	if (!y || !X || !dimX || !Kmax || !nFit || !nLeaf) return NULL;
	size_t const n = (size_t) dimX[0];
	size_t const p = (size_t) dimX[1];
	size_t const k_max = (size_t) *Kmax;
	size_t const n_fit = (size_t) *nFit;
	size_t const n_leaf = (size_t) *nLeaf;
	size_t i, j;
	tree_t *tree = (tree_t *) calloc(1, sizeof(tree_t));
	size_t *idx = (size_t *) malloc(n * p * sizeof(size_t));
	size_t *rnk = (size_t *) malloc(n * p * sizeof(size_t));
	double *err = (double *) malloc(n * sizeof(double));
	double *raw_pred = (double *) calloc(n, sizeof(double));
	size_t *ivec = seq_len(n);
	double *tmpX = (double *) malloc(n * p * sizeof(double));
	if (tree && ivec && idx && rnk && err && raw_pred && tmpX && dimX[0] > *nFit && dimX[0] > *nLeaf * 2 && *nLeaf > 2) {
		tree->p = p;
		tree->xC = (double *) calloc(p, sizeof(double));
		double *y0 = center(y, n, &tree->B, &tree->C);
		if (y0 && tree->xC) {
			memcpy(tmpX, X, p * n * sizeof(double));
			memcpy(err, y0, n * sizeof(double));
			for (j = 0; j < p; j++) {
				sort_reals_widnr(&tmpX[j * n], &idx[j * n], &rnk[j * n], n);
				if (centering) {
					i = j * n + (n >> 1);
					tree->xC[j] = X[j * n + idx[i]];
					tree->xC[j] += X[j * n + idx[i + (size_t) !(n & 1)]];
					tree->xC[j] *= 0.5;
					for (i = 0; i < n; i++) {
						tmpX[j * n + i] = X[j * n + i] - tree->xC[j];
					}
				}
			}
			tree->T = tree_build(raw_pred, y0, err, tmpX, n, &p, 0, idx, rnk, ivec, &k_max, &n_fit, &n_leaf, &tree->B, &n, &clamping);
			free(y0);
		}
		#ifdef DEBUG
		printf("res <- c(%g, ", err[0]);
		for (i = 1; i < n - 1; i++) {
			printf("%g, ", err[i]);
		}
		printf("%g)\n\n", err[i]);
		#endif
		free(err);
		free(ivec);
		free(idx);
		free(rnk);
		free(raw_pred);
		free(tmpX);
	}
	return tree;
}

/**
 * @brief Function to compute the PILOT predictions
 * 
 * @param T Pointer to a root node of a PILOT model
 * @param x Pointer to a vector of features
 * @param p Pointer to the number of variables (i.e., the length of `x`)
 * @param centering Boolean values. If `false`, the values in `x` are already centered. If `true`, the values of `x` are then centered.
 * @param clamping_y Boolean value. If true activate clamping of predicted values
 * @param clamping_x Boolean value. If true activate clamping of covariates
 * @return double
 */
double pilot_predict(tree_t const *T, double const *x, int const *p, 
                     bool const centering, bool const clamping_y, bool const clamping_x) {
	double pred = 0.0;
	double tmpx, pred_add;
	
	if (T && x && p) {
		if (T->p != (size_t) *p) return T->C;
		double const B = T->B;
		node *Tpt = T->T;
		double *xvec = (double *) malloc(*p * sizeof(double));
		/* Center input values */
		if (xvec) {
			if (T->xC && centering) {
				for (size_t j = 0; j < T->p; j++) xvec[j] = x[j] - T->xC[j];
			}
			else {
				memcpy(xvec, x, *p * sizeof(double));
			}
			while(Tpt) {
				tmpx = xvec[Tpt->mod.best_x];
				if (clamping_x) tmpx = clamp(tmpx, Tpt->mod.range[0], Tpt->mod.range[1]);
				if (tmpx < Tpt->mod.best_pivot) {
					pred_add = Tpt->mod.lm_l.con_par + Tpt->mod.lm_l.lin_par * tmpx;
					if (clamping_y) {
						pred = clamp(pred + pred_add, -1.5 * B, B * 1.5);
					}
					else {
						pred += pred_add;
					}
					Tpt = Tpt->l;
				}
				else {
					pred_add = Tpt->mod.lm_r.con_par + Tpt->mod.lm_r.lin_par * tmpx;
					if (clamping_y) {
						pred = clamp(pred + pred_add, -1.5 * B, B * 1.5);
					}
					else {
						pred += pred_add;
					}
					Tpt = Tpt->r;
				}
			}
			free(xvec);
		}
		pred += T->C;
	}
	return pred;
}

/* Memory management functions for the PILOT model */

void free_node(node *nd) {
    if (nd->l) free_node(nd->l);
    if (nd->r) free_node(nd->r);
    if (nd) free(nd);
}

void free_tree(tree_t *T) {
	if (T->T) free_node(T->T);
	if (T->xC) free(T->xC);
	if(T) free(T);
}

#ifdef DEBUG
/*
int main() {
	double x[] = {0.1, 1.02, 2.01, 3.24, 4.25};
	double B, C;
	double *y = center(x, 5, &B, &C);
	for (int i = 0; i < 5; i++){
		printf("x[%d] = %f, y[%d] = %f\n", i, x[i], i, y[i]);
	}
	printf("B = %f, C = %f\n", B, C);
	free(y);
	return 0;
}
*/

#include "../../../.data/nonlinear_data.h"

#define MYCENTER true
#define MYCLAMP true
#define MYCLAMP_Y true
#define MYCLAMP_X true

int main() {
	int Kmax = 33;//(int) (log2((double) nobs) * 0.5);
	int nLeaf = 7;
	int nFit = nLeaf * 2 + 1;
	int dimX[2] = {(int) nobs, (int) nvar};
	double xtest[nvar];
	double pred, obs;
	size_t i, j;
		printf("# Testing PILOT model (at max depth %d)\n", Kmax);
		printf("# ---\n");
	tree_t *myPILOT = pilot(y, x, dimX, &Kmax, &nFit, &nLeaf, MYCENTER, MYCLAMP);
	if (myPILOT) {
		printf("err <- c(");
		for (i = 0; i < nobs - 1; i++) {
			obs = y[i];
			for (j = 0; j < nvar; j++) {
				xtest[j] = x[nobs * j + i];
			}
			pred = pilot_predict(myPILOT, xtest, &dimX[1], MYCENTER, MYCLAMP_Y, MYCLAMP_X);
			/* printf("\tObserved: %g\n", obs);
			printf("\tPredicted: %g\n", pred);
			printf("---\n");
			printf("\tResiduals: %g\n", obs - pred); */
			printf("%g, ", obs - pred);
		}
		obs = y[i];
		for (j = 0; j < nvar; j++) {
			xtest[j] = x[nobs * j + i];
		}
		pred = pilot_predict(myPILOT, xtest, &dimX[1], MYCENTER, MYCLAMP_Y, MYCLAMP_X);
		printf("%g)\n", obs - pred);
		printf("x11()\nhist(res, breaks = 50)\n");
		printf("x11()\nhist(err, breaks = 50)\n");
	}
	free_tree(myPILOT);
	return 0;
}

#endif


