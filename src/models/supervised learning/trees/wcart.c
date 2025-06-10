#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define svoda(a) { if (__builtin_expect((a) != NULL, 1)) free(a); }

typedef struct node {
	size_t v;
	double split;
	double mean;
	double var;
	size_t n;
	struct node *left;
	struct node *right;
} node;

typedef struct wtdat {
	double y;
	double w;
	double *x;
} wtdat;

typedef struct best_split {
	size_t v;
	size_t i;
	double split;
	double ml;
	double sl;
	double mr;
	double sr;
} best_split;

wtdat *dt;
size_t vp_dt = 0;

/**
 * @brief Memory allocation of a node structure
 *
 * @return Pointer to the memory address of an node structure
 */
static inline node * alloc_node(void) {
	node *n = (node *) calloc(1, sizeof(node));
	if (__builtin_expect(n != NULL, 1)) n->var = (double) INFINITY;
	return n;
}

/**
 * @brief Recursively free the memory allocated for a node an its branches
 */
extern void free_node(node *n) {
	if (__builtin_expect(n != NULL, 1)) {
		free_node(n->left);
		free_node(n->right);
		free(n);
	}
}

/**
 * @brief Free the copied data used to fit the model
 *
 * @param n Number of data points
 */
static inline void free_data(size_t n) {
	size_t i;
	if (__builtin_expect(dt && n, 1)) {
		for (i = 0; i < n; i++) svoda(dt[i].x);
		free(dt);
	}
} 

/**
 * @brief Compare data points for a given pre-set variable
 *
 * @param aa Pointer to the first record to compare
 * @param bb Pointer to the second record to compare
 *
 * @return int
 */
static int cmp_data(void const *aa, void const *bb) {
	size_t a = *(size_t *) aa;
	size_t b = *(size_t *) bb;
	if (__builtin_expect(dt == NULL, 0)) return 0;
	if (__builtin_expect(!dt[a].x || !dt[b].x, 0)) return 0; 
	if (__builtin_expect(isnan(dt[a].x[vp_dt]), 0)) return -1;
	if (__builtin_expect(isnan(dt[b].x[vp_dt]), 0)) return 1;
	return 2 * (int) (dt[a].x[vp_dt] > dt[b].x[vp_dt]) - 1;
}

/**
 * @brief Find the best split (i.e., variable selection and splitting point)
 *
 * @param idx Pointer to a vector of indices of data within a node
 * @param n Number of data points in the data
 * @param p Number of features in the data
 * @param phi Minimum number of data points in a terminal leaf
 */
static inline best_split find_best_split(size_t *idx, size_t n, size_t p, size_t phi) {
	size_t j, i;
	double sml, smr, ssl, ssr;
	double ml, mr, sl, sr;
	double cnl, cnr, nrm, tmp;
	best_split ans = {0, 0, nan(""), 0.0, 0.0, 0.0, 0.0};
	if (__builtin_expect(idx == NULL, 0)) return ans;
	double best_fit = (double) INFINITY;
	for (j = 0; j < p; j++) {
		vp_dt = j;
		qsort(idx, n, sizeof(size_t), cmp_data);
		cnl = cnr = sml = smr = ssl = ssr = 0.0;
		for (i = 0; i < phi; i++) {
			tmp = dt[idx[i]].y * dt[idx[i]].w;
			sml += tmp;
			ssl += tmp * dt[idx[i]].y;
			cnl += dt[idx[i]].w;
		}
		nrm = 1.0 / cnl;
		ml = sml * nrm;
		sl = sml * nrm;
		sl -= ml * ml;
		for (; i < n; i++) {
			tmp = dt[idx[i]].y * dt[idx[i]].w;
			smr += tmp;
			ssr += tmp * dt[idx[i]].y;
			cnr += dt[idx[i]].w;
		}
		nrm = 1.0 / cnr;
		mr = smr * nrm;
		sr = smr * nrm;
		sr -= mr * mr;
		/* If best split, then save it */
		if (sl + sr < best_fit) {
			best_fit = sl + sr;
			ans.v = j;
			ans.i = idx[phi];
			ans.split = dt[ans.i].x[j];
			ans.ml = ml;
			ans.sl = sl;
			ans.mr = mr;
			ans.sr = sr;
		}
		i = phi;
		while (i < n - phi) {
			tmp = dt[idx[i]].y * dt[idx[i]].w;
			sml += tmp;
			smr -= tmp;
			tmp *= dt[idx[i]].y;
			ssl += tmp;
			ssr -= tmp;
			cnl += dt[idx[i]].w;
			cnl -= dt[idx[i]].w;
			nrm = 1.0 / cnl;
			ml = sml * nrm;
			sl = sml * nrm;
			sl -= ml * ml;
			nrm = 1.0 / cnr;
			mr = smr * nrm;
			sr = smr * nrm;
			sr -= mr * mr;
			/* If best split, then save it */
			if (sl + sr < best_fit) {
				best_fit = sl + sr;
				ans.v = j;
				ans.i = idx[i + 1];
				ans.split = dt[ans.i].x[j];
				ans.ml = ml;
				ans.sl = sl;
				ans.mr = mr;
				ans.sr = sr;
			}
			i++;
		}
	}
	ans.sl = (double) ans.i / (double) (ans.i - 1);
	ans.sr = (double) (n - ans.i) / (double) (n - ans.i - 1);
	/* Sort dataset with respect to the best */
	vp_dt = ans.v;
	qsort(idx, n, sizeof(size_t), cmp_data);
	return ans;
}


/**
 * @brief Fit a node and splits the node in two branches if necessary
 *
 * @param nd Pointer to a node to train
 * @param idx Pointer to a set of indices to sort the dataset
 * @param n Number of data points to process in the node
 * @param p Number of variables in the data
 * @param K Maximum depth that a terminal node can reach
 * @param k Current depth of the node being processed
 * @param phi Minimum number of data points allowed in a terminal leaf
 */
static void fit_node(node *nd, size_t *idx, size_t n, size_t p, uint8_t K, uint8_t k, size_t phi, double rel_var) {
	size_t i;
	double tmp;
	double my1 = 0.0, my2 = 0.0, nrm = 0.0;
	best_split bs;
	if (__builtin_expect(nd && n >= (phi >> 1) && p && K >= k && phi >= 2 && \
				rel_var > 0.0 && rel_var < 1.0, 1)) {
		if (__builtin_expect(k == 0, 0)) {
			for (i = 0; i < n; i++) {
				nrm += dt[idx[i]].w;
				tmp = dt[idx[i]].y * dt[idx[i]].w;
				my1 += tmp;
				my2 += dt[idx[i]].y * tmp;
			}
			nrm = 1.0 / nrm;
			my1 *= nrm;
			my2 *= nrm;
			my2 -= my1 * my1;
			my2 *= (double) n / (double) (n - 1);
			nd->mean = my1;
			nd->var = my2;
			nd->n = n;
		}
		if (__builtin_expect(k < K && n>= (phi << 1), 1)) { /* Split */
			bs = find_best_split(idx, n, p, phi);
			//if (__builtin_expect((bs.sr + bs.sl) / nd->var < 1.0 - rel_var, 0)) goto terminal_node; // Early stopping rule */
			nd->v = bs.v;
			nd->split = bs.split;
			nd->left = alloc_node(); 
			nd->left->mean = bs.ml;
			nd->left->var = bs.sl;
			nd->right = alloc_node(); 
			nd->right->mean = bs.mr;
			nd->right->var = bs.sr;		
			fit_node(nd->left, idx, bs.i, p, K, k + 1, phi, rel_var);
			fit_node(nd->right, &idx[bs.i], n - bs.i, p, K, k + 1, phi, rel_var);
		}
		else { /* Terminal leaf */
//terminal_node:
			for (i = 0; i < n; i++) {
				nrm += dt[idx[i]].w;
				tmp = dt[idx[i]].y * dt[idx[i]].w;
				my1 += tmp;
				my2 += dt[idx[i]].y * tmp;
			}
			nrm = 1.0 / nrm;
			my1 *= nrm;
			my2 *= nrm;
			my2 -= my1 * my1;
			my2 *= (double) n / (double) (n - 1);
			nd->mean = my1;
			nd->var = my2;
			nd->n = n;
		}
	}
}

/**
 * @brief function to train a weighted CART
 *
 * @param y Pointer to a vector of responses
 * @param w Pointer to a vector of weights
 * @param x Pointer to a matrix of data
 * @param cmf Boolean value. If true, the matrix `x` is stored in column-major format
 * @param n Number of data points (i.e., length of `w` and `y`)
 * @param p Number of features stored in `x`
 * @param max_depth Maximum depth that a terminal leave can have
 * @param phi Minimum number of data points in a terminal leaf
 * @param rel_var Number in (0, 1) used as a stopping rule based on deviance
 *
 * @return Pointer to the root node of a trained weighted CART
 */
extern node * train_wcart(double *y, double *w, double *x, bool cmf, size_t n, size_t p, uint8_t max_depth, size_t phi, double rel_var) {
	node *root = NULL;
	size_t *idx = NULL;
	size_t i, j;
	if (__builtin_expect(y && w && x && n && p && max_depth && phi && \
				rel_var > 0.0 && rel_var < 1.0, 1)) {
		dt = (wtdat *) calloc(n, sizeof(wtdat));
		idx = (size_t *) calloc(n, sizeof(size_t));
		root = alloc_node();
		if (__builtin_expect(root && dt && idx, 1)) {
			/* Copy data in a data structure */
			for (i = 0; i < n; i++) {
				idx[i] = i;
				dt[i].y = y[i];
				dt[i].w = w[i];
				dt[i].x = (double *) calloc(p, sizeof(double));
				if (__builtin_expect(dt[i].x != NULL, 1)) {
					if (cmf) {
						for (j = 0; j < p; j++)
							dt[i].x[j] = x[n * j + i];
					}
					else {
						memcpy(dt[i].x, x, p * sizeof(double));
					}
				}
			}
			/* Weighted CART training */
			fit_node(root, idx, n, p, max_depth, 0, phi, rel_var);
		}
	}
	svoda(idx);
	free_data(n);
	return root;
}

/**
 * @brief Predict the output of a WCART model 
 *
 * @param pred Pointer to a vector where to store the prediction values 
 * @param se Pointer to a vector where to store the standard errors of predicted values
 * @param x Pointer to a matrix of input features
 * @param cmf Boolean value. If true, the matrix `x` is stored in a column-major format
 * @param n Number of data points in `x`
 * @param p Number of features in `x`
 * @param model Pointer to a previously trained model (via the `train_wcart()` function)
 */
extern void predict_wcart(double *pred, double *se, double *x, bool cmf, size_t n, size_t p, node *model) {
	size_t i, j;
	node T;
	bool godeep;
	if (pred && se && x && n && p) {
		dt = (wtdat *) calloc(n, sizeof(wtdat));
		/* Copy data in a data structure */
                for (i = 0; i < n; i++) {
			dt[i].x = (double *) calloc(p, sizeof(double));
                        if (__builtin_expect(dt[i].x != NULL, 1)) {
                                if (cmf) {
                                        for (j = 0; j < p; j++)
                                                dt[i].x[j] = x[n * j + i];
				}
                                else {
                                        memcpy(dt[i].x, x, p * sizeof(double));
                                }
                        }
                }
		/* Compute predictions and their standard errors */
		for (i = 0; i < n; i++) {
			godeep = true;
			T = *model;
			while (godeep) {
				j = T.v;
				/** NOTE: This function assumes that all values in `x` are finite */
				if (dt[i].x[j] < T.split) {
					if (__builtin_expect(T.left != NULL, 1)) {
						T = *T.left;
						printf("left ");
					}
					else {
						pred[i] = T.mean;
						se[i] = T.var; //sqrt(T.var);
						godeep = false;
						printf("\n");
					}
				} 
				else {
					if (__builtin_expect(T.right != NULL, 1)) {
						T = *T.right;
						printf("right ");
					}
					else {
						pred[i] = T.mean;
						se[i] = T.var; //sqrt(T.var);
						godeep = false;
						printf("\n");
					}
				}
			}
		}
		free_data(n);
	}
}

#ifdef DEBUG
#define NT 1000
#define NP 10
#define D 50

int main(void) {
	size_t i = 0;
	double wt[NT] = {0};
	double x[NT * D] = {0};
	double y[NT] = {0};
	double xp[NP * D] = {0};
	double yp[NP] = {0};
	double sp[NP] = {0};
	bool cmf = false;
	node *tree;
	/* Preparing training data set */
	wt[i] = ldexp((double) arc4random(), -32);
	for (i = 0; i < NT; i++) {
		wt[i] = ldexp((double) arc4random(), -32);
	}
	for (i = 0; i < NT * D; i++) x[i] = ldexp((double) arc4random(), -31) - 1.0;
	for (i = 0; i < NT; i++) {
		wt[i] *= (double) NT;
		y[i] = 0.5 * exp(-pow(x[i * D] - 3.0, 2.0) * 0.3);
		y[i] += 1.5 * exp(-pow(x[i * D + 1] + 3.0, 2.0) * 0.3);
		y[i] += sin(2.0 + x[i * D + 2] * 3.0);
		y[i] += 2.0 * fabs(cos(-1.0 + x[i * D + 3] * 0.3));
		y[i] += 1.5 * tanh(y[i] - 1.25 * x[i * D + 4] * x[i * D]) + 5.0 * x[i * D + 3];
	}
	/* Preparing testing data set */
	for (i = 0; i < NP * D; i++) xp[i] = ldexp((double) arc4random(), -31) - 1.0;
	/* Testing the functions implemented above */
	tree = train_wcart(y, wt, x, cmf, NT, D, 5, 3, 0.001);
	predict_wcart(yp, sp, xp, cmf, NP, D, tree);
	free_node(tree);
	/* Print results */
	for (i = 0; i < NP; i++) {
		printf("%.3f (%.3f)\n", yp[i], sp[i]);
	}
	return 0;
}
#endif

