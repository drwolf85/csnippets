#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Modifiable definitions */
#define SRT_MAX_RND_TESTS 10
#define SRT_MIN_SIZE_LEAF 2

/* DO NOT MODIFY THE DEFINITIONS BELOW */
#define SRT_MAX_DEPTH 63

/**
 * @brief Structure of a tree node
 */
typedef struct node {
  struct node *l, *r;
  double split;
  double y_bar;
  double var_y_bar;
} node;

/**
 * @brief Structure of a simple record with one output and one input variable
 */
typedef struct simple_record {
	double y;
	double x;
	uint8_t leaf;
} datum;

/**
 * @brief Comparison function for the values in an input variable
 * 
 * @param aa Void pointer to a `datum` structure
 * @param bb Void pointer to another `datum` structure
 * @return an integer value for the comparison
 */
static int cmp_xval(void const *aa, void const *bb) {
  datum a = *(datum *) aa;
  datum b = *(datum *) bb;
  return 2 * (int) (a.x > b.x) - 1;
}

/**
 * @brief Comparison function for the values of two leaves
 * 
 * @param aa Void pointer to a `datum` structure
 * @param bb Void pointer to another `datum` structure
 * @return an integer value for the comparison
 */
static int cmp_leaf(void const *aa, void const *bb) {
  datum a = *(datum *) aa;
  datum b = *(datum *) bb;
  return 2 * (int) (a.leaf > b.leaf) - 1;
}

/**
 * @brief Recursively free the memory used by a node
 * 
 * @param nd Pointer to a `node` structure
 */
static void free_node(node *nd) {
  if (nd->l) free_node(nd->l);
  if (nd->r) free_node(nd->r);
  free(nd);
}

/**
 * @brief Properly allocate the memory used by a node
 * 
 * @return A pointer to a `node` structure
 */
static inline node * alloc_node(void) {
  node *pt_nd = (node *) calloc(1, sizeof(node));
  return pt_nd;
}

/**
 * @brief Total Sum of Squares and Update Node Predictive Mean
 * 
 * @param T Pointer to a node of tree
 * @param dta Pointer to a set of data used to train the model
 * @param n Number of data points in the training set
 * 
 * @return double
 */
static inline double tss(node *T, datum *dta, uint64_t n) {
  double fm = 0.0, sm = 0.0, res = nan("");
  uint64_t i;
  if (dta && n > 0) {
    for(i = 0; i < n; i++) {
      fm += dta[i].y;
      sm += dta[i].y * dta[i].y;
    }
    T->y_bar = fm / (double) n;
    res = sm - fm * T->y_bar;
  }
  return res;
}

/**
 * @brief Sum of Within Groups Sum of Squares
 * 
 * @param dta Pointer to a set of data used to train the model
 * @param s Splitting value
 * @param n Number of data points in the training set
 * 
 * @return double
 */
static inline double wss(datum *dta, double s, uint64_t n) {
  double fml = 0.0, sml = 0.0, tml = 0.0;
  double fmr = 0.0, smr = 0.0, tmr = 0.0;
  double tmp, res = nan("");
  uint64_t i;
  if (dta && n > 0) {
    for(i = 0; i < n; i++) {
      tmp = (double) (dta[i].x <= s);
      tml += tmp;
      fml += dta[i].y * tmp;
      sml += dta[i].y * dta[i].y * tmp;
      tmp = 1.0 - tmp;
      tmr += tmp;
      fmr += dta[i].y * tmp;
      smr += dta[i].y * dta[i].y * tmp;
    }
     /* printf("\tn.tot: %llu\tn.left: %.0f\t n.right: %.0f", n, tml, tmr); */
    res = sml - fml * fml / (double) tml;
    res += smr - fmr * fmr / (double) tmr;
  }
  /* printf("\twss: %f\n", res); */
  return res;
}

/**
 * @brief Recursively build a tree using a stochastic training algorithm
 * 
 * @param T Pointer to the node that will be splitted
 * @param dta Pointer to a `datum` structure (training data)
 * @param lev Current depth level 
 * @param n Number of samples to train the model within the current node
 * @param max_depth Pointer to the maximum depth of the tree
 * @param mnsn Pointer to minimum number of samples allowed in final leaves 
 */
static void build_tree(node *T, datum *dta, uint64_t n, uint8_t lev, uint8_t *max_depth, uint8_t *mnsn) {
  uint64_t i;
  uint64_t nmn, nmx, nl, tmp = 0;
  double const imxrnd = 1.0 / (1.0 + (double) RAND_MAX);
  double s, tss_in_node, wss_in_node;

  /* Compute predictive values and total sums of squares for the current node */
  tss_in_node = tss(T, dta, n);
  T->var_y_bar = tss_in_node / (double) (n - 1);
  /* Checking if stopping criteria are met */
  if (n >= (((uint64_t) *mnsn) << 1ULL) && lev < *max_depth) {
    /* Check if there are enough data points on the lower and upper ends of the sample */
    nmn = ((uint64_t) *mnsn - 1ULL) % n;
    nmx = (n - ((uint64_t) *mnsn)) % n;
    if (nmn >= nmx) return; 
    /* Randomly find a split to split the data (i.e., method based on random partitioning) */
    T->split = dta[nmn].x + (dta[nmx].x - dta[nmn].x) * 0.5;
    tss_in_node = wss(dta, T->split, n);
    for (i = 0; i < SRT_MAX_RND_TESTS; i++) {
      s = dta[nmn].x + (dta[nmx].x - dta[nmn].x) * ((0.5 + (double) rand()) * imxrnd);
      wss_in_node = wss(dta, s, n);
      T->split += (double) (wss_in_node < tss_in_node) * (s - T->split);
      tss_in_node += (double) (wss_in_node < tss_in_node) * wss_in_node;
    }
    /* Get statistics for the split */
    for (nl = 0, i = 0; i < n; i++, nl += tmp) {
      tmp = (uint64_t) (dta[i].x <= T->split);
      dta[i].leaf = (uint8_t) (tmp == 0ULL);
      dta[i].y -= T->y_bar;
    }
    /* Split the node and build the next level */
    qsort(dta, n, sizeof(datum), cmp_leaf);
    T->l = alloc_node();
    T->r = alloc_node();
    lev++;
    build_tree(T->r, &dta[nl], n - nl, lev, max_depth, mnsn);
    build_tree(T->l, dta, nl, lev, max_depth, mnsn);
  }
}

/**
 * @brief Simple Regression Tree (based on one covariate and one response variable)
 * 
 * @param x Pointer to the values of the covariate 
 * @param y Pointer to the values of the response variable
 * @param n Number of samples to train the model
 * @param max_depth Maximum depth of the tree
 * @param mnsn Minimum number of samples allowed in final leaves 
 * 
 * @return A pointer to a `node` structure containing a tree fully trained
 */
extern node * srt(double *x, double *y, uint64_t n, uint8_t max_depth, uint8_t mnsn) {
  node *T = alloc_node();
  datum *dta = (datum *) calloc(n, sizeof(datum));
  uint64_t i;
  if (T && dta) {
    if (max_depth > SRT_MAX_DEPTH) max_depth = SRT_MAX_DEPTH;
    if (mnsn < SRT_MIN_SIZE_LEAF) mnsn = SRT_MIN_SIZE_LEAF;
    #pragma omp parallel for simd private(i)
    for (i = 0; i < n; i++) {
      dta[i].x = x[i];
      dta[i].y = y[i];
    }
    qsort(dta, n, sizeof(datum), cmp_xval);
    build_tree(T, dta, n, 0, &max_depth, &mnsn);
  }
  free(dta);
  return T;
}

/**
 * @brief Predict using a Simple Regression Tree (S.R.T.)
 * 
 * @param x Value of the covariate used to generate the prediction
 * @param T Pointer to the root `node` of a fully-trained tree
 * 
 * @return double
 */
extern double srt_predict(double x, node *T) {
  double res = nan("");
  if (T) {
    res = T->y_bar;
    if (x > T->split) {
        if (T->r) res += srt_predict(x, T->r);
    }
    else {
        if (T->l) res += srt_predict(x, T->l);
    }
  }
  return res;
}

/**
 * @brief Variance of the prediction made using a Simple Regression Tree (S.R.T.)
 * 
 * @param x Value of the covariate used to generate the prediction
 * @param T Pointer to the root `node` of a fully-trained tree
 * 
 * @return double
 */
extern double srt_var_prediction(double x, node *T) {
  double res = nan("");
  if (T) {
    res = T->var_y_bar;
    if (x > T->split) {
        if (T->r) res = srt_var_prediction(x, T->r);
    }
    else {
        if (T->l) res = srt_var_prediction(x, T->l);
    }
  }
  return res;
}

#ifdef DEBUG
/* Code to the debug the functions above*/

#define _MY_N_ 10000
#define _MY_MAX_DEPTH_ 20
#define _MY_LEAF_SIZE_ 1

double my_test_fun(double x) {
  return 1.0 / M_PI + M_PI * x - cos(2.0 * M_PI * M_PI * x);
}

int main() {
  int i;
  node *mytree;
  double y[_MY_N_];
  double x[_MY_N_];
  double ptx, pred, sep;
  srand(time(NULL));
  for (i = 0; i < _MY_N_; i++) {
    x[i] = 0.5 + (double) i;
    x[i] /= 1.0 + (double) _MY_N_;
    x[i] = 2.0 * x[i] - 1.0;
    y[i] = my_test_fun(x[i]);
  }
  printf("Data are now initilized.\n");
  mytree = srt(x, y, _MY_N_, _MY_MAX_DEPTH_, _MY_LEAF_SIZE_);
  printf("Training of a CART model terminated.\n");
  if (mytree) {
    ptx = 0.54321;
    pred = srt_predict(ptx, mytree);
    sep = sqrt(srt_var_prediction(ptx, mytree));
    printf("Testing prediction in %f: %f (%f)\n", ptx, pred, sep);
    printf("True value in %f: %f\n", ptx, my_test_fun(ptx));
  }
  free_node(mytree);
  printf("Allocated memory is now free.\n");
  return 0;
}
#endif

