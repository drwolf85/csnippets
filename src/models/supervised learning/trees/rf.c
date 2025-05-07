#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Modifiable definitions */
#define RF_MAX_RND_TESTS 10

/* DO NOT MODIFY THE DEFINITIONS BELOW */
#define RF_MIN_SIZE_LEAF 2
#define RF_MAX_DEPTH 63
#define RF_UNIF01 ((0.5 + (double) rand()) * imxrnd)

static uint64_t whv = 0ULL;
static double const imxrnd = 1.0 / (1.0 + (double) RAND_MAX);

/**
 * @brief Structure of a tree node
 */
typedef struct node {
  struct node *l, *r;
  uint64_t var2split;
  double split;
  double y_bar;
  double var_y_bar;
} node;

/**
 * @brief Structure of a simple record with one output and one input variable
 */
typedef struct simple_record {
	double y;
	double *x;
	uint8_t leaf;
} datum;

/**
 * @brief Structure of a vector component with index
 */
typedef struct vector_component {
	double v;
	uint64_t i;
} vec;

/**
 * @brief Comparison function for the values in the output variable
 * 
 * @param aa Void pointer to a `datum` structure
 * @param bb Void pointer to another `datum` structure
 * @return an integer value for the comparison
 */
static int cmp_yval(void const *aa, void const *bb) {
  datum a = *(datum *) aa;
  datum b = *(datum *) bb;
  return 2 * (int) (a.y > b.y) - 1;
}

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
  return 2 * (int) (a.x[whv] > b.x[whv]) - 1;
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
 * @brief Comparison function for the vector components with position indices
 * 
 * @param aa Void pointer to a `vec` structure
 * @param bb Void pointer to another `vec` structure
 * @return an integer value for the comparison
 */
static int cmp_vec(void const *aa, void const *bb) {
  int res;
  vec a = *(vec *) aa;
  vec b = *(vec *) bb;
  res = (int) (a.v > b.v) - (int) (a.v < b.v);
  res += (int) (a.v == b.v) * (2 * (int) (a.i > b.i) - 1);
  return res;
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

extern void free_forest(node **T, uint64_t nt) {
  uint64_t t;
  for (t = 0; t < nt; t++) free_node(T[t]);
  free(T);
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
      tmp = (double) (dta[i].x[whv] <= s);
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
 * @param n Number of samples to train the model within the current node
 * @param lev Current depth level 
 * @param p Pointer to the number of predictor variables
 * @param max_depth Pointer to the maximum depth of the tree
 * @param mnsn Pointer to minimum number of samples allowed in final leaves 
 */
static void build_tree(node *T, datum *dta, uint64_t n, uint8_t lev, uint64_t *p, uint8_t *max_depth, uint8_t *mnsn) {
  uint64_t i, j;
  uint64_t nmn, nmx, nl, tmp = 0;
  uint64_t mxvar_check = (*p >> 1);
  double s, tss_in_node, wss_in_node;
  vec *spread;
  
  /* Compute predictive values and total sums of squares for the current node */
  tss_in_node = tss(T, dta, n);
  T->var_y_bar = tss_in_node / (double) (n - 1);
  /* Checking if stopping criteria are met */
  if (n >= (((uint64_t) *mnsn) << 1ULL) && lev < *max_depth) {
    /* Check if there are enough data points on the lower and upper ends of the sample */
    nmn = ((uint64_t) *mnsn - 1ULL) % n;
    nmx = (n - ((uint64_t) *mnsn)) % n;
    if (nmn >= nmx) return; 
    /* Pick a variable based on its "sorted spread" */
    whv = rand() % *p;
    qsort(dta, n, sizeof(datum), cmp_xval);
    /* Randomly find a split to split the data (i.e., method based on random partitioning) */
    T->var2split = whv;
    T->split = dta[nmn].x[whv] + (dta[nmx].x[whv] - dta[nmn].x[whv]) * 0.5;
    tss_in_node = wss(dta, T->split, n);
    for (i = 0; i < RF_MAX_RND_TESTS; i++) {
    /*for (i = nmn + 1; i < nmx; i++) { */
      s = dta[nmn].x[whv] + (dta[nmx].x[whv] - dta[nmn].x[whv]) * RF_UNIF01;
      /*s = dta[i].x[whv];*/
      wss_in_node = wss(dta, s, n);
      T->split += (double) (wss_in_node < tss_in_node) * (s - T->split);
      tss_in_node += (double) (wss_in_node < tss_in_node) * (wss_in_node - tss_in_node);
    }
    /* Get leaf indices and compute residuals for the next split */
    for (nl = 0, i = 0; i < n; i++, nl += tmp) {
      tmp = (uint64_t) (dta[i].x[T->var2split] <= T->split);
      dta[i].leaf = (uint8_t) (tmp == 0ULL);
      dta[i].y -= T->y_bar;
    }
    /* Split the node and build the next level */
    qsort(dta, n, sizeof(datum), cmp_leaf);
    T->l = alloc_node();
    T->r = alloc_node();
    lev++;
    build_tree(T->r, &dta[nl], n - nl, lev, p, max_depth, mnsn);
    build_tree(T->l, dta, nl, lev, p, max_depth, mnsn);
  }
}

/**
 * @brief Random Forests based on Regression Tree (based on one covariate and one response variable)
 * 
 * @param x Pointer to a set of the covariates (an n-by-p matrix)
 * @param y Pointer to the values of the response variable
 * @param n Number of samples to train the model
 * @param p Number of predictor variables
 * @param nt Number of trees to grow
 * @param max_depth Maximum depth of the tree
 * @param mnsn Minimum number of samples allowed in final leaves 
 * @param X_Col_Major Boolean value. It must be set to `true` if the matrix in `x` is stored in column major format
 * 
 * @return A pointer to a `node` structure containing a tree fully trained
 */
extern node ** rf(double *x, double *y, uint64_t n, uint64_t p, uint64_t nt, uint8_t max_depth, uint8_t mnsn, bool X_Col_Major) {
  uint64_t i, j, t;
  double *Xmat_t;
  node **T = (node **) calloc(nt, sizeof(node *));
  datum *dta = (datum *) calloc(n, sizeof(datum));
  if (max_depth > RF_MAX_DEPTH) max_depth = RF_MAX_DEPTH;
  if (mnsn < RF_MIN_SIZE_LEAF) mnsn = RF_MIN_SIZE_LEAF;
  Xmat_t = (double *) malloc(p * n * sizeof(double));
  if (T && dta && Xmat_t) {
    if (X_Col_Major) {
      #pragma omp parallel for simd private(i, j) collapse(2)
      for (i = 0; i < n; i++) for (j = 0; j < p; j++) Xmat_t[p * i + j] = x[j * n + i];
    }
    else {
      #pragma omp parallel for simd private(i)
      for (i = 0; i < n * p; i++) Xmat_t[i] = x[i];
    }
    #pragma omp parallel for simd private(i)
    for (i = 0; i < n; i++) {
      dta[i].x = &Xmat_t[i * p];
      dta[i].y = y[i];
    }
    for (t = 0; t < nt; t++) {
      T[t] = alloc_node();
      if (T[t]) build_tree(T[t], dta, n, 0, &p, &max_depth, &mnsn);
    }
  }
  free(Xmat_t);
  free(dta);
  return T;
}

/**
 * @brief Predict using a Regression Tree (R.T.)
 * 
 * @param x Pointer to the vector of covariates used to generate the prediction
 * @param T Pointer to the root `node` of a fully-trained tree
 * 
 * @return double
 */
static double rt_predict(double *x, node *T) {
  double res = nan("");
  if (x && T) {
    res = T->y_bar;
    if (x[T->var2split] > T->split) {
        if (T->r) res += rt_predict(x, T->r);
    }
    else {
        if (T->l) res += rt_predict(x, T->l);
    }
  }
  return res;
}

/**
 * @brief Variance of the prediction made using a Regression Tree (R.T.)
 * 
 * @param x Pointer to the vector of covariates used to generate the prediction
 * @param T Pointer to the root `node` of a fully-trained tree
 * 
 * @return double
 */
static double rt_var_prediction(double *x, node *T) {
  double res = nan("");
  if (T) {
    res = T->var_y_bar;
    if (x[T->var2split] > T->split) {
        if (T->r) res = rt_var_prediction(x, T->r);
    }
    else {
        if (T->l) res = rt_var_prediction(x, T->l);
    }
  }
  return res;
}

/**
 * @brief Predict using a Random Forest (R.F.)
 * 
 * @param x Pointer to the vector of covariates used to generate the prediction
 * @param T Pointer to pointer of a fully-trained tree
 * @param nt Number of trees that grew
 * 
 * @return double
 */
extern double rf_predict(double *x, node **T, uint64_t nt) {
  double res = nan("");
  double summary = 0.0;
  double wt, nrm = 0.0;
  uint64_t t;
  if (x && T) {
    for (t = 0; t < nt; t++) {
      wt = rt_var_prediction(x, T[t]);
      wt = 1.0 / sqrt(wt + 1e-9);
      nrm += wt;
      summary += wt * rt_predict(x, T[t]);
    }
    nrm = 1.0 / nrm;
    res = summary * nrm;
  }
  return res;
}

/**
 * @brief Variance of the prediction made using a Random Forest (R.F.)
 * 
 * @param x Pointer to the vector of covariates used to generate the prediction
 * @param T Pointer to pointer of a fully-trained tree
 * @param nt Number of trees that grew
 * 
 * @return double
 */
extern double rf_var_prediction(double *x, node **T, uint64_t nt) {
  double res = nan("");
  double summary = 0.0;
  double wt, nrm = 0.0;
  uint64_t t;
  if (x && T) {
    for (t = 0; t < nt; t++) {
      wt = rt_var_prediction(x, T[t]);
      wt = 1.0 / sqrt(wt + 1e-9);
      nrm += wt;
      summary += wt * rt_var_prediction(x, T[t]);
    }
    nrm = 1.0 / nrm;
    res = summary * nrm;
  }
  return res;

}

#ifdef DEBUG
/* Code to the debug the functions above*/

#define _MY_N_ 10000
#define _MY_MAX_DEPTH_ 12
#define _MY_LEAF_SIZE_ 5
#define _MY_N_TREES_ 100

double my_test_fun(double *x) {
  return 1.0 / M_PI + M_PI * x[0] - cos(2.0 * M_PI * M_PI * x[1]) + x[2] / (0.125 + fabs(x[2]));
}

int main() {
  int i, j;
  node **myforest;
  double y[_MY_N_];
  double x[_MY_N_ * 3];
  double ptx[3], pred, sep;
  srand(time(NULL));
  for (i = 0; i < _MY_N_; i++) {
    x[3 * i] = 0.5 + (double) i;
    x[3 * i] /= 1.0 + (double) _MY_N_;
    x[3 * i] = 2.0 * x[3 * i] - 1.0;
    for (j = 1; j < 3; j++) x[3 * i + j] = 2.0 * RF_UNIF01 - 1.0;
    y[i] = my_test_fun(&x[3 * i]);
  }
  printf("Data are now initilized.\n");
  myforest = rf(x, y, _MY_N_, 3, _MY_N_TREES_, _MY_MAX_DEPTH_, _MY_LEAF_SIZE_, false);
  printf("Training of a Random Forest model terminated.\n");
  if (myforest) {
    ptx[0] = 0.54321;
    ptx[1] = -0.54321;
    ptx[2] = 0.25;
    pred = rf_predict(ptx, myforest, _MY_N_TREES_);
    sep = sqrt(rf_var_prediction(ptx, myforest, _MY_N_TREES_));
    printf("Testing prediction in (");
    for (j = 0; j < 3; j++) printf("%f%s", ptx[j], j + 1 < 3 ? ", " : ")");
    printf(": %f (%f)\n", ptx, pred, sep);
    printf("True value: %f\n", my_test_fun(ptx));
  }
  free_forest(myforest, _MY_N_TREES_);
  printf("Allocated memory is now free.\n");
  return 0;
}
#endif

