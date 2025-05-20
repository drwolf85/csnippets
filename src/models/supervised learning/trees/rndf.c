#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef struct data {
  double *x;
  size_t dx;
  double *y;
  size_t dy;
  double pj;
  size_t i;
} data;

typedef struct node {
  double *prj;
  double spl;
  size_t nl;
  struct node *lf;
  double *pred;
} node;

/**
 * Allocate nodes of a tree
 *
 * @n_nodes number of nodes
 *
 * @return pointer to a node structure
 */
static inline node * alloc_nodes(size_t n_nodes) {
  node *res = (node *) calloc(n_nodes, sizeof(node));
  return res;
}

/**
 * Free branch as if it is the root of a tree
 *
 * @nd Pointer to a node structure
 * @n_nodes Pointer to the number of nodes to free
 */
static inline void free_root(node *nd, size_t n_nodes) {
  size_t i = 0;
  if (__builtin_expect(nd != NULL, 1)) {
    for (; i < n_nodes; i++) {
      if (__builtin_expect(nd[i].prj != NULL, 1)) free(nd[i].prj);
      if (__builtin_expect(nd[i].lf != NULL, 1)) free_root(nd[i].lf, nd[i].nl);
      if (__builtin_expect(nd[i].pred != NULL, 1)) free(nd[i].pred);
    }
    free(nd);
  }
}

/**
 * Pseudo-random numbers normally distributed
 *
 * @mu Mean of the distribution
 * @sd Standard deviation of the distribution
 */
static inline double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = arc4random();
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

/**
 * Pseudo-random generation of a projection vector
 *
 * d Number of dimensions
 */
static inline double * rproj(size_t d) {
  size_t i = 0;
  double const isd = d > 0 ? fabs(1.0 / sqrt((double) d)) : 0.0;
  double *v = (double *) calloc(d, sizeof(double));
  if (__builtin_expect(v != NULL, 1)) {
    for (; i < d; i++) {
      v[i] = rnorm(0.0, isd);
    }
  }
  return v;
}

/**
 * One-dimensional projections for one data point
 *
 * @param dt Pointer to a data point
 * @param p Pointer to a `dx`-dimensional projection vector
 * @param dx Number of features in input
 */
static inline double proj2dbl(data *dt, double *p, size_t dx) {
  double res = 0.0;
  size_t i, nf = 0;
  if (__builtin_expect(dt && p, 1)) {
    for (i = 0; i < dx; i++) {
      if (__builtin_expect(isfinite(dt->x[i]), 1)) {
        res += dt->x[i] * p[i];
        nf++;
      }
    }
    if (__builtin_expect(nf > 0 && nf < dx, 0)) {
      res *= (double) dx / (double) nf;
    }
  }
  return res;
}

/**
 * Set one-dimensional projections on the data structure
 *
 * @param dt Pointer to a data structure
 * @param p Pointer to a `dx`-dimensional projection vector
 * @param n Number of elements in the data structure
 * @param dx Number of features in input
 */
static inline void proj1d(data *dt, double *p, size_t n, size_t dx) {
  size_t i, j, nf;
  if (__builtin_expect(dt && p, 1)) {
    for (i = 0; i < n; i++) {
      if (__builtin_expect(dt[i].x && dt[i].y, 1)) {
	dt[i].pj = 0.0;
	nf = 0;
	for (j = 0; j < dx; j++) {
	  if (__builtin_expect(isfinite(dt[i].x[j]), 1)) {
	    dt[i].pj += dt[i].x[j] * p[j];
	    nf++;
	  }
	}
	if (__builtin_expect(nf > 0 && nf < dx, 0)) {
	  dt[i].pj *= (double) dx / (double) nf;
	}
      }
    }
  }
}

/**
 * Compare projections
 *
 * @aa Pointer to the first object to compare
 * @bb Pointer to the second object to compare
 *
 * @return int
 */
static int cmp_prj(void const *aa, void const *bb) {
  data a = *(data *) aa;
  data b = *(data *) bb;
  return 2 * (a.pj > b.pj) - 1;
}

/**
 * Generate a pseudo-random number from a triangular distribution 
 *
 * @param mu location parameter of the triangular distribution
 * @param sd scale parameter of the triangular distribution
 *
 * @return double
 */
static inline double rtriang(double mu, double sd) {
   uint64_t u, v;
   double a, b;
   u = v = arc4random();
   u <<= 32ULL;
   u |= arc4random();
   v |= (u << 32ULL);
   a = ldexp((double) u, -64);
   b = ldexp((double) v, -64);
   return (a - b) * sd + mu;
}

/**
 * Generate a pseudo-random number to split the projected values
 *
 * @param dt Pointer to a structure of data
 * @param n Number of data point in the branch to split
 * @param min_leaf Minimum number of points in the terminal leaf
 * @param cnt Pointer to the count number of units less than the split
 *
 * @return double
 */
static inline double rsplit(data *dt, size_t n, size_t min_leaf, size_t *cnt) {
  double a, rs = nan("");
  if (__builtin_expect(dt != NULL, 1)) {
    qsort(dt, n, sizeof(data), cmp_prj);
    rs = rtriang((double) (n >> 1), 0.5 * (double) (n - (min_leaf << 1)));
    a = rs - floor(rs);
    *cnt = (size_t) ceil(rs);
    rs = dt[(size_t) floor(rs)].pj * (1.0 - a) + dt[*cnt].pj * a;
  }
  return rs;
}

/**
 * Free data structure
 *
 * @param dt Pointer to the data structure to free
 * @param n Size of the preallocated structure
 */
static inline void free_data(data *dt, size_t n) {
  size_t i;
  if (__builtin_expect(dt != NULL, 1)) {
    for (i = 0; i < n; i++) {
	    if (dt[i].x) free(dt[i].x);
	    if (dt[i].y) free(dt[i].y);
    }
    free(dt);
  }
}

/**
 * Copy data into their proper data structure
 *
 * @param n Number of data points to train a random forest model
 * @param X Pointer to the input data
 * @param dx Number of variables/features in input
 * @param Y Pointer to the output data
 * @param dy Number of response dimensions in output
 * @param cmf Boolean value. If `true`, the matrice `X` and `Y` are stored in
 *            a Column-Major Format (CMF)
 *
 * @return Pointer to a data structure of well-organized data
 */
static inline data * cpdt(size_t n, double *X, size_t dx, double *Y, size_t dy, bool cmf) {
  size_t i, j, nc = 0;
  data *dt = calloc(n, sizeof(data));
  if (__builtin_expect(dt && X && Y, 1)) {
    for (i = 0; i < n; i++) {
      dt[i].x = calloc(dx, sizeof(double));
      dt[i].y = calloc(dy, sizeof(double));
      dt[i].i = i;
      if (dt[i].x && dt[i].y) { 
	nc++;
	if (cmf) {
	  for (j = 0; j < dx; j++) dt[i].x[j] = X[n * j + i];
	  for (j = 0; j < dy; j++) dt[i].y[j] = Y[n * j + i];
        }
        else {
          memcpy(dt[i].x, &X[dx * i], dx * sizeof(double));
          memcpy(dt[i].y, &Y[dy * i], dy * sizeof(double));
        }
      }
    }
  }
  if (__builtin_expect(nc < n, 0)) free_data(dt, n);
  return dt;
}

/* Build a randomized tree
 *
 * @param root Pointer to a root/branch/leaf node
 * @param dt Pointer to a data structure
 * @param n Number of data points
 * @param dx Number of features in input
 * @param dy Number of features in output
 * @param depth Depth of the node in the tree (NB: leaves have zero depth)
 * @param min_leaf Minimum number in a terminal node
 */
extern void rnd_tree(node *root, data *dt, size_t n, size_t dx, size_t dy, size_t depth, size_t min_leaf)  {
  size_t i, cnt;
  double inv = 1.0;
  if (__builtin_expect(dt && root, 1)) {
    if (__builtin_expect(n >= (min_leaf << 1) && depth > 0, 0)) { /* Split further */
      root->prj = rproj(dx); /* Random projection */
      proj1d(dt, root->prj, n, dx);
      root->spl = rsplit(dt, n, min_leaf, &cnt); /* Draw the split along the projection */
      root->nl = 2;
      root->lf = alloc_nodes(root->nl);
      depth--; /* Perform the "churn of the data" */
      rnd_tree(&root->lf[0], dt, cnt, dx, dy, depth, min_leaf);
      rnd_tree(&root->lf[1], &dt[cnt], n - cnt, dx, dy, depth, min_leaf);
    }
    else { /* Make a terminal node */
      root->pred = (double *) calloc(dy, sizeof(double));
      if (__builtin_expect(root->pred != NULL, 1)) {
        for (cnt = 0; cnt < n; cnt++) {
	  for (i = 0; i < dy; i++) {
	    root->pred[i] += dt[cnt].y[i];
	  }
        }
        inv /= (double) n;
#ifdef DEBUG
	printf("Prd RTM: ");
#endif
        for (i = 0; i < dy; i++) {
          root->pred[i] *= inv;
#ifdef DEBUG
	  printf("%s%.2f ", root->pred[i] >= 0.0 ? " " : "", root->pred[i]);
#endif
        }
#ifdef DEBUG
	printf("\n");
#endif
      }
    }
  }
}

/**
 * Random forest prediction algorithm
 *
 * @param root Pointer to an array of node structures
 * @param nt Number of trees stored in the array `root`
 * @param dt Pointer to a structure of data
 * @param n Number of data points
 * @param dx Number of input variables
 * @param dy Number of output variables
 */
extern void rf_predict(node *root, size_t nt, data *dt, size_t n, size_t dx, size_t dy) {
  size_t t, i, j;
  double pj;
  double invnt = 1.0 / (double) (nt > 0 ? nt : 1.0);
  node *T;
  if (__builtin_expect(root && dt, 1)) {
    for (i = 0; i < n; i++) {
      memset(dt[i].y, 0, dy * sizeof(double));
      for (t = 0; t < nt; t++) {
        T = &root[t];
        while (__builtin_expect(T->nl > 0, 1)) {
	  pj = proj2dbl(&dt[i], T->prj, dx);
	  T = pj <= T->spl ? &T->lf[0] : &T->lf[1];
        }
	if (__builtin_expect(T->pred != NULL, 1)) {
	  for (j = 0; j < dy; j++) {
	    dt[i].y[j] += T->pred[j];
	  }
	}
      }
      for (j = 0; j < dy; j++) dt[i].y[j] *= invnt;
    }
  }
}

#ifdef DEBUG
int main(void) {
  size_t const D = 3;
  size_t const K = 24;
  size_t const N = 7;
  size_t const ML = 3;
  size_t i, j;
  double Y[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  double X[] = {-1.080221, 2.13662, -0.4679458, 0.1231624, -0.2964475, -0.1006749, -0.6403836, -0.2791455, 1.111347, 1.156517, 1.703992, 0.653971, 1.222196, -0.4262778, 1.374594, 0.4969235, -1.120565, 1.819362, -0.3283002, -0.4837726, -1.248508};
  /* Testing the random projection generator */
  double *test = rproj(D);
  double rsp = nan("");
  data *dt = NULL;
  node *roots = NULL;
  if (test) {
    printf("Rnd Prj: ");
    for (i = 0; i < D; i++) {
      printf("%s%f ", test[i] >= 0.0 ? " " : "", test[i]);
    }
    printf("\n");
    /* Testing Random Triangular Number Generator at 64bit */
    printf("Rnd Tri: ");
    for (i = 0; i < D; i++) {
      test[i] = rtriang(0.5, 2.0);
      printf("%s%f ", test[i] >= 0.0 ? " " : "", test[i]);
    }
    printf("\n");
    /* Testing data copying procedure */
    dt = cpdt(N, X, D, Y, D, true);
    free_data(dt, N);
    dt = cpdt(N, X, D, Y, D, false);
    if (dt) {
      /* Testing 1D projections */
      proj1d(dt, test, N, D);
      rsp = rsplit(dt, N, ML, &i);
      printf("Spl Cnt: %lu\n", i);
      printf("D1D RPj: ");
      for (i = 0; i < N; i++) {
	printf("%s%f ", dt[i].pj >= 0.0 ? " " : "", dt[i].pj);
      }
      printf("\n");
      printf("Rnd SPL: %f\n", rsp);
      /* Building random tree for test */
      printf("\n");
      roots = alloc_nodes(D);
      if (roots) {
	for (i = 0; i < D; i++) { /* This loop builds a forest */
          rnd_tree(&roots[i], dt, N, D, D, K, ML);
	  printf("\n");
	}
	/* Testing random forest predictions */
	printf("Before predicting\n");
	for (i = 0; i < N; i++) {
	  printf("Prd (%lu): ", i);
	  for (j = 0; j < D; j++) {
	    printf("%s%.2f ", dt[i].y[j] >= 0.0 ? " " : "", dt[i].y[j]);
	  }
	  printf("\n");
	}
        rf_predict(roots, D, dt, N, D, D);
	printf("After predicting\n");
	for (i = 0; i < N; i++) {
	  printf("Prd (%lu): ", i);
	  for (j = 0; j < D; j++) {
	    printf("%s%.2f ", dt[i].y[j] >= 0.0 ? " " : "", dt[i].y[j]);
	  }
	  printf("\n");
	}
      }
      if (roots) free_root(roots, D);
    }
    if (dt) free_data(dt, N);
  }
  if (test) free(test);
  return 0;
}
#endif

