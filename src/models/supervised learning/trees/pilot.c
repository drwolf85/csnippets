/**
 * @brief Fast Linear Model Trees by PIecewise Linear Organic Tree (PILOT)
 *        based on the work of Raymaekers, Rousseeuw, Verdonck, and Yao 
 *        published on February 7, 2023.
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
    res += (double) (x > up) * (up - res);
    res += (double) (x < lw) * (lw - res);
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
		mx += (double) (mx < x[i]) * (x[i] - mx);
		mn += (double) (x[i] < mn) * (x[i] - mn);
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
 * @param n length of the vector above (size_t)
 */
static void sort_reals_wid(double *x, size_t *idx, size_t n) {
    double *s, *y;
    uint8_t c;
    uint64_t v;
    size_t i, h[256], ch[256], *sid;
    sid = (size_t *) malloc(n * sizeof(size_t));
    s = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));

    if (s && y && sid && x && idx) {
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
 * @param idx Pointer to a vector of the ranks of variable `x`
 * @param n Length of `ivec`
 * @param x Pointer to a variable `x`
 * @param y Pointer to a vector of residuals
 * @param n_leaf Pointer to the minimal number of cases in a terminal leaf
 * @param j Index of the variable to process
 */
void BIC_select(leaf_model *mod, size_t *ivec, size_t *idx, size_t n, double *x, double *y, size_t const *n_leaf, size_t j) {
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

	/* Compute sufficient statistics */
	for (i = 0; i < *n_leaf; i++) {
		sx[0] += x[idx[ivec[i]]];
		sy[0] += y[idx[ivec[i]]];
		sxx[0] += x[idx[ivec[i]]] * x[idx[ivec[i]]];
		sxy[0] += y[idx[ivec[i]]] * x[idx[ivec[i]]];
		syy[0] += y[idx[ivec[i]]] * y[idx[ivec[i]]];
	}
	for (; i < n; i++) {
		sx[1] += x[idx[ivec[i]]];
		sy[1] += y[idx[ivec[i]]];
		sxx[1] += x[idx[ivec[i]]] * x[idx[ivec[i]]];
		sxy[1] += y[idx[ivec[i]]] * x[idx[ivec[i]]];
		syy[1] += y[idx[ivec[i]]] * y[idx[ivec[i]]];
	}
	/* Compute parameter estimates and BIC for constant */
	par[0] = par[2] = ninv * (sy[0] + sy[1]);
	bic = (double) n * log((syy[0] + syy[1]) * ninv - par[0] * par[0]) + log((double) n); /* BIC_CON */
	if (bic < mod->best_bic) {
		mod->best_model = CON;
		mod->best_x = j;
		mod->best_pivot = INFINITY;
		mod->lm_l.con_par = par[0];
		mod->best_bic = bic;
	}
	/* Compute parameter estimates and BIC for linear */
	par[0] = sx[0] + sx[1];
	par[1] = (sxy[0] + sxy[1] - par[0] * par[2]) / (sxx[0] + sxx[1] - par[0] * par[0] * ninv);
	par[0] *= ninv * par[1];
	par[0] = par[2] - par[0];
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
		mod->lm_r.con_par = par[2];
		mod->best_bic = bic;
	}
	/* Compute parameter estimates and BIC for broken linear */
	par[1] = (sxy[0] - sx[0] * par[0]) / (sxx[0] - sx[0] * sx[0] * inl);
	par[2] = (sxy[1] - sx[1] * par[2]) / (sxx[1] - sx[1] * sx[1] * inmnl);
	par[0] = (sy[0] + sy[1]) * ninv - par[1] * sx[0] * inl - par[2] * sx[1] * inmnl;
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
		sx[0]  += x[idx[ivec[*n_leaf + i]]];
		sy[0]  += y[idx[ivec[*n_leaf + i]]];
		sxx[0] += x[idx[ivec[*n_leaf + i]]] * x[idx[ivec[*n_leaf + i]]];
		sxy[0] += y[idx[ivec[*n_leaf + i]]] * x[idx[ivec[*n_leaf + i]]];
		syy[0] += y[idx[ivec[*n_leaf + i]]] * y[idx[ivec[*n_leaf + i]]];
		sx[1]  -= x[idx[ivec[*n_leaf + i]]];
		sy[1]  -= y[idx[ivec[*n_leaf + i]]];
		sxx[1] -= x[idx[ivec[*n_leaf + i]]] * x[idx[ivec[*n_leaf + i]]];
		sxy[1] -= y[idx[ivec[*n_leaf + i]]] * x[idx[ivec[*n_leaf + i]]];
		syy[1] -= y[idx[ivec[*n_leaf + i]]] * y[idx[ivec[*n_leaf + i]]];
		inl = 1.0 / (double) (*n_leaf + i);
		inmnl = 1.0 / (double) (n - *n_leaf - i);
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
		/* Compute parameter estimates and BIC for broken linear */
		par[1] = (sxy[0] - sx[0] * par[0]) / (sxx[0] - sx[0] * sx[0] * inl);
		par[2] = (sxy[1] - sx[1] * par[2]) / (sxx[1] - sx[1] * sx[1] * inmnl);
		par[0] = (sy[0] + sy[1]) * ninv - par[1] * sx[0] * inl - par[2] * sx[1] * inmnl;
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
}

/**
 * @brief Function to select a model by screening all variables in `X`
 * 
 * @param err Pointer to a vector of residuals
 * @param X Pointer to a matrix of covariates
 * @param n Number of cases to process (i.e., length of `ivec`)
 * @param p Pointer to the number of variables in `X`
 * @param idx Pointer of indices based on the order of the values in `X`
 * @param ivec Pointer to the cases to process for the newly allocated node 
 * @param n_leaf Pointer to the minimal number of cases in a terminal leaf
 * @param N Pointer to the total number of cases in `X`
 * 
 * @return Pointer to a newly allocated tree node
 */
node * model_select(double *err, double *X, size_t const n, size_t const *p, size_t *idx, size_t *ivec, 
		            size_t const *n_leaf, size_t const *N) {
	node *best = (node *) calloc(1, sizeof(node));
	size_t j;
	if (best) {
		best->mod.best_bic = INFINITY;
		for(j = 0; j < *p; j++) {
			BIC_select(&best->mod, ivec, &idx[*N * j], n, &X[*N * j], err, n_leaf, j);
		}
		j = *N * best->mod.best_x;
		best->mod.range[0] = X[j + idx[j + ivec[0]]];
		best->mod.range[1] = X[j + idx[j + ivec[n - 1]]];
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
 * @param idx Pointer to a matrix of indices based on the values in `X`
 * @param N Total number of rows in `X` and `idx`
 * 
 * @return size_t (Location of the first index store for the right branch)
 */
size_t fix_ivec(size_t *ivec, size_t n, node *T, double *X, size_t *idx, size_t const *N) {
	size_t i, elk, ell, end = n - 1;
	double xl, xk;
	bool unswapped;
	i = 0;
	while(i < end) {
		unswapped = true;
		elk = idx[*N * T->mod.best_x + ivec[i]];
		xl = X[*N * T->mod.best_x + elk];
		if (xl >= T->mod.best_pivot) {
			do {
				ell = idx[*N * T->mod.best_x + ivec[end]];
				xk = X[*N * T->mod.best_x + ell];
				if (xk < T->mod.best_pivot) {
					ivec[i] ^= ivec[end];
					ivec[end] ^= ivec[i];
					ivec[i] ^= ivec[end];
					unswapped = false;
				}
				--end;
			} while(unswapped && end > i);
		}
		i += (size_t) (end > i);
	}
	return i;
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
 * @param idx Pointer to a matrix of indices store according to the order of variables in `X`
 * @param ivec Pointer to a vector of indices related to extract proper cases for the training
 * @param k_max Pointer to the maximal depth allowed
 * @param n_fit Pointer to the minimal number of cases used to fit a model
 * @param n_leaf Pointer to the minimal number of cases in terminal leaves
 * @param B Pointer to a value acting as a bound
 * @param N Pointer to the total number of cases in `X` and `idx`
 * 
 * @return A pointer to a node/branch structure
 */
node * tree_build(double *raw_pred, double *y, double *err, double *X, size_t const n, size_t const *p, size_t K, size_t *idx, size_t *ivec, 
		  size_t const *k_max, size_t const *n_fit, size_t const *n_leaf, double const *B, size_t const *N) {
	node *T = NULL;
	double x;
	size_t i, ell;
	if (K < *k_max && n >= *n_fit && *n_leaf >= 2 && *n_fit > *n_leaf * 2) {
		T = model_select(err, X, n, p, idx, ivec, n_leaf, N);
		T->depth = K;
		K += (size_t) (T->mod.best_model > 1);
		if (T->mod.best_model == CON) {
			return T;
		}
		else if (T->mod.best_model == LIN) {
			for (i = 0; i < n; i++) { /* Predictions of the linear model */
				ell = idx[*N * T->mod.best_x + ivec[i]];
				x = X[*N * T->mod.best_x + ell];
				raw_pred[ell]  = T->mod.lm_l.con_par;
				raw_pred[ell] += T->mod.lm_l.lin_par * x;
				/* Residual update */
				err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], -*B * (-3.0), *B * 3.0);
			}
			T->l = tree_build(raw_pred, y, err, X, n, p, K, idx, ivec, k_max, n_fit, n_leaf, B, N);
		}
		else { /* Processing Threshold models for growing branches*/
			if (T->mod.best_model == PCON) { /* Prediction of piecewise constants */
				for (i = 0; i < n; i++) {
					ell = idx[*N * T->mod.best_x + ivec[i]];
					x = X[*N * T->mod.best_x + ell];
					raw_pred[ell]  = T->mod.lm_l.con_par * (double) (x <  T->mod.best_pivot);
					raw_pred[ell] += T->mod.lm_r.con_par * (double) (x >= T->mod.best_pivot);
					/* Residual update */
					err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], *B * (-3.0), *B * 3.0);
				}
			}
			else { /* Predictions of broken and piecewise linear models */
				for (i = 0; i < n; i++) {
					ell = idx[*N * T->mod.best_x + ivec[i]];
					x = X[*N * T->mod.best_x + ell];
					raw_pred[ell]  = (T->mod.lm_l.con_par + T->mod.lm_l.lin_par * x) * (double) (x <  T->mod.best_pivot);
					raw_pred[ell] += (T->mod.lm_r.con_par + T->mod.lm_r.lin_par * x) * (double) (x >= T->mod.best_pivot);
					/* Residual update */
					err[ell] = y[ell] - clamp(y[ell] - err[ell] + raw_pred[ell], *B * (-3.0), *B * 3.0);
				}
			}
			ell = fix_ivec(ivec, n, T, X, idx, N);
			qsort(ivec, ell, sizeof(size_t), cmp_ivec);
			qsort(&ivec[ell], n - ell, sizeof(size_t), cmp_ivec);
			T->l = tree_build(raw_pred, y, err, X, ell, p, K, idx, ivec, k_max, n_fit, n_leaf, B, N);
			T->r = tree_build(raw_pred, y, err, X, n - ell, p, K, idx, &ivec[ell], k_max, n_fit, n_leaf, B, N);
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
 * @return A Pointer to a tree structure with the fin
 */
tree_t * pilot(double *y, double *X, int *dimX, int *Kmax, int *nFit, int *nLeaf) {
	if (!y || !X ||! dimX) return NULL;
	size_t const n = (size_t) dimX[0];
	size_t const p = (size_t) dimX[1];
	size_t const k_max = (size_t) *Kmax;
	size_t const n_fit = (size_t) *nFit;
	size_t const n_leaf = (size_t) *nLeaf;
	size_t j;
	tree_t *tree = (tree_t *) calloc(1, sizeof(tree_t));
	size_t *idx = (size_t *) malloc(n * p * sizeof(size_t));
	double *err = (double *) malloc(n * sizeof(double));
	double *raw_pred = (double *) calloc(n, sizeof(double));
	size_t *ivec = seq_len(n);
	if (tree && ivec && idx && err && raw_pred) {
		double *y0 = center(y, n, &tree->B, &tree->C);
		if (y0) {
			memcpy(err, y0, n * sizeof(double));
			for (j = 0; j < p; j++) sort_reals_wid(&X[j * n], &idx[j * n], n);
			tree->T = tree_build(raw_pred, y0, err, X, n, &p, 0, idx, ivec, &k_max, &n_fit, &n_leaf, &tree->B, &n);
		}
		free(y0);
	}
	free(err);
	free(ivec);
	free(idx);
	free(raw_pred);
	return tree;
}

/**
 * @brief Function to compute the PILOT predictions
 * 
 * @param T Pointer to a root node of a PILOT model
 * @param x Pointer to a vector of features
 * @param p Pointer to the number of variables (i.e., the length of `x`)
 * @return double
 */
double pilot_predict(tree_t const *T, double *x, int const *p) {
	double pred = 0.0;
	double tmpx, pred_add;
	
	if (T && x && p) {
		double const B = T->B;
		node *Tpt = T->T;
		while(Tpt) {
			tmpx = clamp(x[Tpt->mod.best_x], Tpt->mod.range[0], Tpt->mod.range[1]);
			switch (Tpt->mod.best_model) {
			case CON:
				pred_add = Tpt->mod.lm_l.con_par;
				pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
				Tpt = Tpt->l;
				break;
			case LIN:
				pred_add = Tpt->mod.lm_l.con_par + Tpt->mod.lm_l.lin_par * tmpx;
				pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
				Tpt = Tpt->l;
				break;
			case PCON:
				if (tmpx < Tpt->mod.best_pivot) {
					pred_add = Tpt->mod.lm_l.con_par;
					pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
					Tpt = Tpt->l;
				}
				else {
					pred_add = Tpt->mod.lm_r.con_par;
					pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
					Tpt = Tpt->r;
				}
				break;
			default: /* case BLIN and PLIN */
				if (tmpx < Tpt->mod.best_pivot) {
					pred_add = Tpt->mod.lm_l.con_par + Tpt->mod.lm_l.lin_par * tmpx;
					pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
					Tpt = Tpt->l;
				}
				else {
					pred_add = Tpt->mod.lm_r.con_par + Tpt->mod.lm_r.lin_par * tmpx;
					pred = clamp(pred + pred_add, -3.0 * B, B * 3.0);
					Tpt = Tpt->r;
				}
				break;
			}
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

int main() {
	int Kmax = 99;
	int nFit = 5;
	int nLeaf = 2;
	int dimX[2] = {(int) nobs, (int) nvar};
	double xtest[nvar];
	double pred, obs;
	size_t i, j;
	tree_t *myPILOT = pilot(y, x, dimX, &Kmax, &nFit, &nLeaf);
	if (myPILOT) {
		printf("# Testing PILOT model\n");
		printf("---\nc(");
		for (i = 0; i < nobs - 1; i++) {
			obs = y[i];
			for (j = 0; j < nvar; j++) {
				xtest[j] = x[nobs * j + i];
			}
			pred = pilot_predict(myPILOT, xtest, &dimX[1]);
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
		pred = pilot_predict(myPILOT, xtest, &dimX[1]);
		printf("%g)\n", obs - pred);
	}
	free_tree(myPILOT);
	return 0;
}

#endif

