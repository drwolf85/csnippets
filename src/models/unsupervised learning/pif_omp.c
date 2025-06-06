/**
 * @title Proximity Isolation Forest Using Single Prototypes
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

void *dt; /* Global pointer for storing a generic structured or unstructured dataset */
double *H; /* Pointer to store normalizing factors */
struct random_data rnd_data; /* Structure for seeding PRNG */
char *statebuf;
#pragma omp threadprivate(rnd_data, statebuf)

typedef struct node {
	void *proto;
	double split;
	uint8_t dep;
	bool type;
	uint64_t size;
	struct node *left;
	struct node *right;
} node;

typedef struct dblvec {
	double v;
	uint64_t i;
} dblvec;

dblvec *idx;
#pragma omp threadprivate(idx)

static inline uint64_t arc64rnd(void) {
	int32_t u, v;
	int64_t r;
	random_r(&rnd_data, &u);
	random_r(&rnd_data, &v);
	r = (((int64_t) u << 32ULL) ^ (int64_t) v);
	return *(uint64_t *) &r; 
}

/**
 * @brief Normalizing factors (i.e., vector of harmonic numbers)
 * 
 * @param n an integer number
 * @return double 
 */
static inline double cfun(uint64_t n) {
    return 2.0 * (H[n - 2] - (double) (n - 1) / (double) n);
}

/**
 * @brief Function to allocate the structure of a node in a tree
 *
 * @return A pointer to a node structure
 */
static inline node * alloc_node() {
	node *n = (node *) calloc(1, sizeof(node));
	return n;
}

/**
 * @brief Function to free the allocated memory of a tree (or branch of a tree)
 *
 * @param n A pointer to a node structure
 * @parma nt Numbers of tree to free
 */
static void free_node(node *n, uint64_t nt) {
	uint64_t i;
	if (n) {
		for (i = 0; i < nt; i++) {
			n[i].proto = NULL;
			n[i].split = 0.0;
			n[i].dep = 0;
			n[i].type = false;
			n[i].size = 0;
			if(n[i].left) free_node(n[i].left, 1);
			if(n[i].right) free_node(n[i].right, 1);
		}
		free(n);
	}
}

/**
 * @brief Random number generator from a Uniform(a, b)
 *
 * @param a Lower bound of the support of the uniform distribution
 * @param b Upper bound of the support of the uniform distribution
 * 
 * @return double
 */
static inline double runif(double a, double b) {
	uint64_t u = arc64rnd();
	double const mn = fmin(a, b);
	double const mx = fmax(a, b);
	return mn + ldexp((double) u, -64) * (mx - mn);
}

/**
 * @brief Comparison function between two items from a `dblvec` structure
 *
 * @param aa Pointer to the memory of the first item to compare
 * @param bb Pointer to the memory of the second item to compare
 *
 * @return int
 */
static int cmp_dblvec(void const *aa, void const *bb) {
  double a = ((dblvec *) aa)->v;
  double b = ((dblvec *) bb)->v;
  return (a > b) * 2 - 1;
}

/**
 * @brief Proximity Isolation Tree with Signle Prototype
 *
 * @param nd Pointer to a `node` structure
 * @param dtsz Size of a datum stored in `*dt`
 * @param idx Pointer to a dblvec structure of length `n`
 * @param n Number of data points in the current node
 * @param k Current Depth of the node in the tree
 * @param l Maximum Depth of the tree
 * @param dst Pointer to a distance (or dissimilarity) function
 */
static void pit_sp(node *nd, uint64_t dtsz, dblvec *idx, 
	 uint64_t n, uint8_t k, uint8_t l,
	 double (*ds)(void const *, void const *)) {
	uint64_t whp, i;
	bool test;
	if (__builtin_expect(nd && idx && n && l && ds, 1)) {
		if (__builtin_expect(k >= l || n <= 1ULL, 0)) {
			nd->size = n;
			nd->type = false;
			nd->dep = k;
		}
		else {
			nd->type = true;
			nd->dep = k;
			whp = arc64rnd() % n;
			nd->proto = (void *) &((uint8_t *) dt)[idx[whp].i * dtsz];
			for (i = 0; i < n; i++)
				idx[i].v = ds(&((uint8_t *) dt)[idx[i].i * dtsz], nd->proto);
			qsort(idx, n, sizeof(dblvec), cmp_dblvec);
			nd->split = runif(idx[0].v, idx[n - 1].v);
			test = true;
			for (i = 0; test && i < n; i++) {
				test = (bool) (idx[i].v <= nd->split);
			}
			nd->left = alloc_node();
			nd->right = alloc_node();
			if (__builtin_expect(nd->left && nd->right, 1)) {
				pit_sp(nd->left, dtsz, idx, i, k + 1, l, ds);
				pit_sp(nd->right, dtsz, &idx[i], n - i, k + 1, l, ds);
			}
			else {
				free_node(nd->left, 1);
				free_node(nd->right, 1);
				nd->proto = NULL;
				nd->type = false;
				nd->split = nan("");
			}
		}
	}
}

/**
 * @brief Proximity Isolation Forest with Signle Prototype
 *
 * @param nt Number of proximity isolation trees with single prototype to train
 * @param dtsz Size of a datum stored in `*dt`
 * @param n Number of data points stored in `*dt`
 * @param subs Number of subsamples to randomly select from the dataset `*dt`
 * @param l Maximum Depth of the tree
 * @param dst Pointer to a distance (or dissimilarity) function
 */
static inline node * train_pif_sp(uint64_t nt, uint64_t dtsz,
	 uint64_t n, uint64_t subs, uint8_t l, 
	 double (*ds)(void const *, void const *)) {
	uint64_t i, j;
	node *roots = (node *) calloc(nt, sizeof(node));
	#pragma omp parallel 
	{
		rnd_data.state = NULL;
		statebuf = (char *) calloc(256, sizeof(char));
		if (__builtin_expect(statebuf != NULL, 1))
			initstate_r(arc4random() ^ omp_get_thread_num(), \
				    statebuf, sizeof(char) * 256, &rnd_data);
		idx = (dblvec *) calloc(subs, sizeof(dblvec));
	}
	if (__builtin_expect(roots && idx && statebuf, 1)) {
		#pragma omp parallel for default(shared) private(i, j)
		for (i = 0; i < nt; i++) {
			for (j = 0; j < subs; j++)  /* Subsampling with replacement */
				idx[j].i = arc64rnd() % n;
			pit_sp(&roots[i], dtsz, idx, subs, 0, l, ds);
		}
	}
	#pragma omp parallel
	{
		if (__builtin_expect(statebuf != NULL, 1)) free(statebuf);
		if (__builtin_expect(idx != NULL, 1)) free(idx);
	}
	return roots;
}

/**
 * @brief Compute the isolation score (or path length)
 * 
 * @param x Pointer to a data point stored in `*dt`
 * @param tr Pointer to a tree in the forest
 * @param e current path length 
 * @param ds Pointer to a distance (or dissimilarity) function
 *
 * @return double Isolation score
 */
static double path_length(void *x, node *tr, uint8_t e,
                          double (*ds)(void const *, void const *)) {
    double res = (double) e;
    double dst;
    if (!tr->type) {
        if (tr->size > 1) res += cfun(tr->size);
        return res;
    }
    else {
        dst = ds(x, tr->proto);
        return path_length(x, dst <= tr->split ? tr->left : tr->right, e + 1, ds);
    }
}

/**
 * @brief Compute the anomaly score of a proximity isolation forest
 * 
 * @param x Index to a data point stored in `*dt`
 * @param sz Size of a datum stored in `*dt`
 * @param forest Pointer to a trained foreset
 * @param nt Number of trees in the forest
 * @param nss Number of subsamples used to construct the trees in the forest
 * @param ds Pointer to a distance (or dissimilarity) function
 *
 * @return double 
 */
static inline double fuzzy_anomaly_score(uint64_t x, uint64_t sz, node *forest, 
                                         uint64_t nt, uint64_t nss, 
                                         double (*ds)(void const *, void const *)) {
	uint64_t i;
	double avglen = 1.0;
	double const nrmc = -1.0 / cfun((uint32_t) nss);
	for (i = 0; i < nt; i++) {
		avglen += log1p(- pow(2.0, \
			path_length((void *) &((uint8_t *) dt)[x * sz], \
			&forest[i], 0, ds) * nrmc));
	}
	avglen /= (double) nt;
	return 1.0 - exp(avglen);
}

/**
 * @brief Anomaly Detection via Proximity Isolation Forest with Signle Prototype
 *
 * @param n Number of data points stored in `info_dat`
 * @param info_dat Pointer to an array of (structured or unstructured) data
 * @param dtsz Size of a datum stored in `info_dat`
 * @param nt Number of proximity isolation trees with single prototype to train
 * @param nss Number of subsamples to randomly select from the dataset `info_dat`
 * @param l Maximum Depth of the tree
 * @param dst Pointer to a distance (or dissimilarity) function
 *
 * return A pointer of a vector with the anomaly scores for each record in `info_dat`
 */
extern double * pif_one(uint64_t n, void *info_dat, uint64_t size_dat,
                        uint64_t nt, uint64_t nss, uint8_t l,
                        double (*ds)(void const *, void const *)) {
	uint64_t i;
	double *res = NULL;
	node *forest = NULL;

	if (__builtin_expect(n && info_dat && size_dat && nt && nss && l, 1)) {
		H = (double *) malloc(nss * sizeof(double));
		res = (double *) malloc(n * sizeof(double));
		dt = info_dat;
		if (__builtin_expect(H && dt && res, 1)) {
			H[0] = 1.0;
			for (i = 1; i < nss; i++)
				H[i] = H[i - 1] + 1.0 / (1.0 + (double) i);
			forest = train_pif_sp(nt, size_dat, n, nss, l, ds);
			if (__builtin_expect(forest != NULL, 1)) {
				#pragma omp parallel for
				for (i = 0; i < n; i++)
					res[i] = fuzzy_anomaly_score(i, \
						size_dat, forest, nt, nss, ds);
			}
			if (__builtin_expect(forest != NULL, 1))
				free_node(forest, nt);
		}
		if (__builtin_expect(H != NULL, 1)) free(H);
	}
	return res;
}

#ifdef DEBUG

#define D 10

double mydiss(void const *aa, void const *bb) {
	uint64_t i;
	double *a = (double *) aa;
	double *b = (double *) bb;
	double s, res = 0.0;
	for (i = 0; i < D; i++) {
		s = a[i] - b[i];
		res += cosh(s) - 1.0;
	}
	return res;
}

#define NOFF 150
#define NGR1 1350
#define NGR2 1500
#define N (NOFF + NGR1 + NGR2)

#define NT 500
#define NSS 30
#define TDEP 4

int main(void) {
	uint64_t i, j;
	double *scores;
	double *dataset = malloc(N * D * sizeof(double));
	statebuf = (char *) calloc(256, sizeof(char));
	rnd_data.state = NULL;
	if (__builtin_expect(statebuf && dataset, 1)) {
		initstate_r(arc4random(), statebuf, \
			    sizeof(char) * 256, &rnd_data);
		for (j = 0, i = 0; i < NOFF * D; i++) ((double *) dataset)[i] = runif(-20.0, 20.0);
		for (j = i, i = 0; i < NGR1 * D; i++, j++) ((double *) dataset)[j] = runif(0.5, 1.0);
		for (i = 0; i < NGR2 * D; i++, j++) ((double *) dataset)[j] = runif(-1.0, -0.5);
		if (__builtin_expect(statebuf != NULL, 1)) free(statebuf);
		scores = pif_one(N, dataset, sizeof(double) * D, NT, NSS, TDEP, mydiss);
		if (__builtin_expect(scores != NULL, 1)) {
			for (i = 0; i < N; i++) {
				printf("%.3f ", scores[i]);
			}
			printf("\n");
		}
		if (__builtin_expect(scores != NULL, 1)) free(scores);
	}
	if (__builtin_expect(dataset != NULL, 1)) free(dt);
	return 0;
}

#endif

