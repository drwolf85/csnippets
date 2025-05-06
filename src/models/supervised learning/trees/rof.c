/**
 * @brief Random Organic Forest (ROF)
 */
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define myfree(x) if((void *)(x)) free((void *)(x));


typedef struct datum {
	double *y;
	int dy;
	double *x;
	int dx;
} datum;

/* Model types */
typedef enum Model_type { CON, /* CONstant. The recursion stops after fitting this model. */
  PCON /* Piecewise CONstant fit, as in CART. */
} model_t;

typedef struct parameters {
    double *con_par; /* Constant predictor (or model intercept) */
} param_t;

typedef struct leaf_model {
    double best_pivot; /* Random pivot value (a.k.a. the splitting value) */
    model_t best_model; /* Best model type */
    double best_bic; /* Best BIC */
    size_t best_x; /* Random predictor */
    param_t lm_l; /* Parameters for the left side */
    param_t lm_r; /* Parameters for the right side */
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
	double *B;
	double *C;
} tree_t;

static void free_node(node *nd) {
    if (nd->l) free_node(nd->l);
    if (nd->r) free_node(nd->r);
    myfree(nd->mod.lm_l.con_par);
    myfree(nd->mod.lm_r.con_par);
    myfree(nd);
}

static void free_trees(tree_t *tree, size_t ntrees) {
	size_t i;
	for (i = 0; i < ntrees; i++)
		if (tree[i].T) free_node(tree[i].T);
	myfree(tree->xC);
	myfree(tree->B);
	myfree(tree->C);
	myfree(tree);
}

static void model_select(leaf_model *mod, int *idx, datum *data, int n, int *nr, int const *n_fit, int const *n_leaf) {
	int i, j, tmp, nl;
	double *sy;
	double *syy;
	double ninv[3] = {0};
	double *pala, *para;
	double bic, cib;
	*nr = 0;
	mod->best_x = (size_t) (rand() % data->dx);
	mod->best_pivot = data[rand() % n].x[mod->best_x];

	mod->lm_l.con_par = (double *) calloc(data->dy, sizeof(double));
	mod->lm_r.con_par = (double *) calloc(data->dy, sizeof(double));

	para = (double *) calloc(data->dy, sizeof(double));
	pala = (double *) calloc(data->dy, sizeof(double));
	
	sy = (double *) calloc(2 * data->dy, sizeof(double));
	syy = (double *) calloc(2 * data->dy, sizeof(double));

	if (sy && syy && para && pala && mod->lm_l.con_par && mod->lm_r.con_par) {
		/* Compute sufficient statistics */
		for (i = 0; i < n; i++) {
			tmp = (int) (data[idx[i]].x[mod->best_x] >= mod->best_pivot);
			*nr += tmp;
			for (j = 0; j < data->dy; j++) {
				sy[tmp * data->dy + j]  += data[idx[i]].y[j];
				syy[tmp * data->dy + j] += data[idx[i]].y[j] * data[idx[i]].y[j];
			}
		}
		nl = n - *nr;
		ninv[0] = 1.0 / (double) (n);
		ninv[1] = 1.0 / (double) (*nr);
		ninv[2] = 1.0 / (double) (nl);

		/* CON */
		bic = 0.0;
		for (j = 0; j < data->dy; j++) {
			pala[j] = ninv[0] * (sy[j] + sy[data->dy + j]);
			cib = (double) n * log((syy[j] + syy[data->dy + j]) * ninv[0] - pala[j] * pala[j]);
			cib += log((double) n);
			bic += cib; /* BIC_CON */
		}
		if (bic < mod->best_bic) {
			mod->best_bic = bic;
			mod->best_model = CON;
			memcpy(mod->lm_l.con_par, pala, data->dy * sizeof(double));
			memcpy(mod->lm_r.con_par, pala, data->dy * sizeof(double));
		}
		/* Random Branching */
		if (*nr >= *n_fit && n - *nr >= *n_fit) {
			/* PCON */
			bic = 0.0;
			for (j = 0; j < data->dy; j++) {
				pala[j] = sy[j] * ninv[2];
				para[j] = sy[data->dy + j] * ninv[1];
				cib = syy[j] * ninv[2] - pala[j] * pala[j];
				cib += syy[data->dy + j] * ninv[1] - para[j] * para[j];
				cib = (double) n * log(cib) + 5.0 * log((double) n); /* BIC_PCON */
				bic += cib;
			}
			if (bic < mod->best_bic) {
				mod->best_bic = bic;
				mod->best_model = PCON;
				memcpy(mod->lm_l.con_par, pala, data->dy * sizeof(double));
				memcpy(mod->lm_r.con_par, para, data->dy * sizeof(double));
			}
		}
	}

	myfree(sy);
	myfree(syy);
	myfree(para);
	myfree(pala);
}

static void separate_idx(int *idx, int n, int nl, int nr, datum *data, leaf_model *mod) {
	bool unswap = true;
	int i, ml = 0, mr = n - 1;
	for (i = 0; i < mr; i += (int) unswap) {
		unswap = true;
		if (data[idx[i]].x[mod->best_x] >= mod->best_pivot) {
			idx[i] ^= idx[mr];
			idx[mr] ^= idx[i];
			idx[i] ^= idx[mr];
			--mr;
			unswap = false;
		}
	}
}

static node * fit_rnd_tree(tree_t const *T, int *idx, double *Y0, datum *data, int n, int K,
                           int *k_max, int const *n_fit, int const *n_leaf,
                           bool const clamping) {
	int nr, nl, i, j;
	double pred;
	node *L = (node *) calloc(1, sizeof(node));
	L->mod.best_bic = INFINITY;
	if (L && K < *k_max && n >= *n_fit && *n_leaf > 2 && *n_fit > *n_leaf * 2) {
		/* Model select */
		model_select(&L->mod, idx, data, n, &nr, n_fit, n_leaf);
		nl = n - nr;
		if (L->mod.best_model == CON) {
			return L;
		}
		else { /* Update training dataset */
			K++;
			/* Residual update (boosting step) */
			for (i = 0; i < n; i++) {
				for (j = 0; j < data->dy; j++) {
					pred = data[idx[i]].x[L->mod.best_x];
					pred = (double)(pred < L->mod.best_pivot) * \
					       (L->mod.lm_l.con_par[j]) + \
					       (double)(pred >= L->mod.best_pivot) * \
					       (L->mod.lm_r.con_par[j]);

					if (clamping) {
						pred = Y0[data->dy * idx[i] + j] - data[idx[i]].y[j] + pred;
						data[idx[i]].y[j] = Y0[data->dy * idx[i] + j] - \
						                    fmax(-1.5 * T->B[j], fmin(pred, 1.5 * T->B[j]));
					}
					else {
						data[idx[i]].y[j] -= pred;
					}
		       		}
		       	}
			separate_idx(idx, n, nl, nr, data, &L->mod);
			L->l = fit_rnd_tree(T, idx, Y0, data, nl, K, k_max, n_fit, n_leaf, clamping);
			L->r = fit_rnd_tree(T, &idx[nl], Y0, data, nr, K, k_max, n_fit, n_leaf, clamping);
		}
	}
	return L;
}

tree_t * rof_fit(double const *Y, int *dimY, double const *X, int *dimX, 
                 int n_trees, int *Kmax, int const *nFit, int const *nLeaf,
                 bool const clamping) {
	int i, j;
	tree_t *forest = NULL;
	datum *data;

	if (*dimY != *dimX) return forest;

	int *idx = (int *) calloc(*dimX, sizeof(int));
	double *Yt = (double *) calloc(dimY[0] * dimY[1], sizeof(double));
	double *Y0 = (double *) calloc(dimY[0] * dimY[1], sizeof(double));
	double *Xt = (double *) calloc(dimX[0] * dimX[1], sizeof(double));
	double *Yb = (double *) calloc(dimY[1], sizeof(double));
	double *Xb = (double *) calloc(dimX[1], sizeof(double));
	double *Yc = (double *) calloc(dimY[1], sizeof(double));
	double *Xc = (double *) calloc(dimX[1], sizeof(double));
	data = (datum *) calloc(*dimX, sizeof(datum));
	forest = (tree_t *) calloc(n_trees, sizeof(tree_t));

	if (forest && idx && Y0 && Yt && Xt && Yb && Xb && Yc && Xc && data) {
		/* Transpose and get min and max of each variable */
		for (j = 0; j < dimY[1]; j++) Yt[j] = Y[*dimY * j];
		for (j = 0; j < dimX[1]; j++) Xt[j] = X[*dimX * j];
		memcpy(Yb, Yt, dimY[1] * sizeof(double));
		memcpy(Yc, Yt, dimY[1] * sizeof(double));
		memcpy(Xb, Xt, dimX[1] * sizeof(double));
		memcpy(Xc, Xt, dimX[1] * sizeof(double));
		data[0].y = Yt;
		data[0].dy = dimY[1];
		data[0].x = Xt;
		data[0].dx = dimX[1];
		for (i = 1; i < *dimX; i++) {
			for (j = 0; j < dimY[1]; j++) {
				Yt[dimY[1] * i + j] = Y[*dimY * j + i];
				Yb[j] = fmax(Yb[j], Yt[dimY[1] * i + j]);
				Yc[j] = fmin(Yc[j], Yt[dimY[1] * i + j]);
			}
			for (j = 0; j < dimX[1]; j++) {
				Xt[dimX[1] * i + j] = X[*dimX * j + i];
				Xb[j] = fmax(Xb[j], Xt[dimX[1] * i + j]);
				Xc[j] = fmin(Xc[j], Xt[dimX[1] * i + j]);
			}
			data[i].y = &Yt[dimY[1] * i];
			data[i].dy = dimY[1];
			data[i].x = &Xt[dimX[1] * i];
			data[i].dx = dimX[1];
		}
		/* Compute B and C */
		for (j = 0; j < dimY[1]; j++) {
			Yb[j] -= Yc[j];
			Yb[j] *= 0.5;
			Yc[j] += Yb[j];
		}
		for (j = 0; j < dimX[1]; j++) {
			Xb[j] -= Xc[j];
			Xb[j] *= 0.5;
			Xc[j] += Xb[j];
		}
		for (i = 0; i < *dimX; i++) {
			for (j = 0; j < dimY[1]; j++) Yt[dimY[1] * i + j] -= Yc[j];
			for (j = 0; j < dimX[1]; j++) Xt[dimX[1] * i + j] -= Xc[j];
		}
		memcpy(Y0, Yt, dimY[0] * dimY[1] * sizeof(double));
		for (i = 0; i < n_trees; i++) {
			for (j = 0; j < dimX[0]; j++) idx[j] = rand() % dimX[0]; /* Bagging */
			forest[i].xC = Xc;
			forest[i].B = Yb;
			forest[i].C = Yc;
			forest[i].p = (size_t) dimX[1];
			forest[i].T = fit_rnd_tree(&forest[i], idx, Y0, data, *dimX, 0, Kmax, nFit, nLeaf, clamping);
			memcpy(Yt, Y0, dimY[0] * dimY[1] * sizeof(double));
		}
	}

	myfree(idx);
	myfree(Xt);
	myfree(Yt);
	myfree(Y0);
	myfree(Xb);
	myfree(data);
	return forest;
}

void rof_predict(double *y, int const dy, 
                 tree_t const *F, int const n_trees, 
                 double const *x, int const dx, bool const clamping) {
	int i, j;
	double tmpx;
	double pred_add;
	double const invnt = 1.0 / (double) n_trees;
	double *pred = (double *) malloc(dy * sizeof(double));
	double *xvec = (double *) malloc(dx * sizeof(double));
	if (pred && y && F && x && xvec) {
		memcpy(y, F->C, dy * sizeof(double));
		for (j = 0; j < dx; j++) xvec[j] = x[j] - F->xC[j];
		for (i = 0; i < n_trees; i++) {
			memset(pred, 0, dy * sizeof(double));
			node *Tpt = F[i].T;
			while(Tpt) {
				tmpx = xvec[Tpt->mod.best_x];
				if (tmpx < Tpt->mod.best_pivot) {
					for (j = 0; j < dy; j++) {
						pred_add = Tpt->mod.lm_l.con_par[j];
						pred[j] += pred_add;
						if (clamping) pred[j] = fmax(-1.5 * F->B[j], fmin(pred[j], 1.5 * F->B[j]));
					}
					Tpt = Tpt->l;
				}
				else {
					for (j = 0; j < dy; j++) {
						pred_add = Tpt->mod.lm_r.con_par[j];
						pred[j] += pred_add;
						if (clamping) pred[j] = fmax(-1.5 * F->B[j], fmin(pred[j], 1.5 * F->B[j]));
					}
					Tpt = Tpt->r;
				}
			}
			for (j = 0; j < dy; j++) {
				y[j] += invnt * pred[j];
			}
		}
	}
	myfree(xvec);
	myfree(pred);
}

#ifdef DEBUG

#include "../../../.data/nonlinear_data.h"

/* Training setting */
#define NTREES 10
#define MYCLAMP true

int main() {
	int Kmax = 33;//(int) (log2((double) nobs) * 0.5);
	int nLeaf = 7;
	int nFit = nLeaf * 2 + 1;
	int dimX[2] = {(int) nobs, (int) nvar};
	int dimY[2] = {(int) nobs, (int) 1};
	double xtest[nvar];
	double pred, obs;
	size_t i, j;

	srand(time(NULL));

	printf("# Testing ROF model (at max depth %d)\n", Kmax);
	printf("# ---\n");

	tree_t *myROF = rof_fit(y, dimY, x, dimX, NTREES, &Kmax, &nFit, &nLeaf, MYCLAMP);
	if (myROF) {
		printf("err <- c(");
		for (i = 0; i < nobs - 1; i++) {
			obs = y[i];
			for (j = 0; j < nvar; j++) {
				xtest[j] = x[nobs * j + i];
			}
			rof_predict(&pred, 1, myROF, NTREES, xtest, dimX[1], MYCLAMP);
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
		rof_predict(&pred, 1, myROF, NTREES, xtest, dimX[1], MYCLAMP);
		printf("%g)\n", obs - pred);
		printf("x11()\nhist(err, breaks = 50)\n");
	}
	free_trees(myROF, NTREES);
	return 0;
}

#endif

