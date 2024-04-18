#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
// #include <omp.h>

/* DIF algorithm using randomized DNN, max cuts and fuzzy scores based on a t-conorm */

/**
 * @brief Deep Neural Network structure
 * 
 */
typedef struct deep_neural_network {
	struct deep_neural_network *child;
	uint32_t n_in;
	uint32_t n_out;
	double complex *coef;
	double complex *bias;
} dnn;

/**
 * @brief Isolation tree structure
 * 
 */
typedef struct iso_tree {
    uint32_t size;
    uint8_t type;
    uint8_t level;
    struct iso_tree *left;
    struct iso_tree *right;
    double *lincon;
    uint32_t nv_lat;
    double threshold;
} iTrees;

/**
 * @brief Structure for isolation models 
 *
 */
typedef struct iso_models {
    iTrees **isoForest;
    dnn **isoNet;
} iModels;

/**
 * @brief Proejction vector
 *
 */
typedef struct vec_proj {
	double v; /* Value */
	uint32_t i; /* Original data index */
} vector;

double *H;
uint32_t *subs;
vector *proj;
// #pragma omp threadprivate(subs, proj)

static inline double complex cs_sign(double complex x) {
    return x / (1.0 + cabs(x));
}
/**
 * @brief Comparison function between the values of structure `vec_proj`
 * 
 * @param aa Pointer to the first element to compare
 * @param bb Pointer to the second element to compare
 * @return in
 */
int cmp_vec(void const *aa, void const *bb) {
    vector a = *(vector *) aa;
    vector b = *(vector *) bb;
    return (int) (a.v > b.v) - (int) (a.v < b.v);
}

/**
 * @brief Normalizing factors (i.e., vector of harmonic numbers)
 * 
 * @param n an integer number
 * @return double 
 */
static inline double cfun(uint32_t n) {
    return 2.0 * (H[n - 2] - (double) (n - 1) / (double) n);
}

/**
 * `runif` generates a random number between `a` and `b` using the `rand` function
 * 
 * @param a lower bound
 * @param b upper bound
 * 
 * @return A random number between a and b.
 */
static inline double runif(double a, double b) {
    unsigned long u, m = ~(1 << 31);
        u = rand() & m;
        return ldexp((double) u, -31) * (b - a) + a;
}

/** 
 * The function rnorm() generates a random number from a normal distribution with
 * mean mu and standard deviation sd
 * 
 * @param mu mean of the normal distribution
 * @param sd standard deviation
 * 
 * @return A random number from a normal distribution with mean mu and standard deviation sd.
 */
static inline double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
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

double dpoisson(uint32_t x, double lambda) {
    double z = nan("");
    if (lambda == 0.0) {
        return (double) (x == 0);
    }
    else if (lambda > 0.0) {
        z = log(lambda) * (double) x;
        z -= lambda; 
        z -= lgamma((double) x + 1.0);
        z = exp(z);
    }
    return z;
}

uint32_t qpoisson(double p, double lambda) {
    uint32_t i, z = 0;
    double tmp;
    if (p >= 0.0 && p <= 1.0 && lambda >= 0.0) {
        tmp = dpoisson(0, lambda);
        for (i = 1; tmp <= p; i++) {
            tmp += dpoisson(i, lambda);
        }
        z = (double) (i - 1);    
    }
    return z;
}

uint32_t rpoisson(double lambda) {
    uint32_t u, m, z = 0;
    if (lambda >= 0.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = qpoisson(ldexp((double) u, -31), lambda);
    }
    return z;
}

/**
 * @brief Neural Network evaluation function 
 *
 * @param x Pointer to input values
 * @param net Pointer to a neural network layer 
 * @param l Number of the layer in input (zero is used for the input layer)
 *
 * @return double * Pointer to an output vector
 */
double complex * net_eval(double complex *x, dnn *net, uint32_t l) {
    uint32_t i, j, pos;
    double complex *tmp_vec = (double complex *) calloc(net->n_out, sizeof(double complex));
    if (net && tmp_vec && x) {
        for (pos = 0, j = 0; j < net->n_out; j++) {
            tmp_vec[j] = net->bias[j];
            for (i = 0; i < net->n_in; i++, pos++)
                tmp_vec[j] += x[i] * net->coef[pos];
            tmp_vec[j] = cs_sign(tmp_vec[j]);
        }
        if (l) free(x);
        if (net->child) {
            return net_eval(tmp_vec, net->child, l + 1);
        }
        else {
            return tmp_vec;
        }
    }
}

static inline void rnd_bias(double complex *bias, double complex **tmp, uint32_t n, uint32_t nv_out) {
    uint32_t i, j;
    double rln, imn, rlx, imx, tmp_db;
    for (j = 0; j < nv_out; j++) {
        rlx = rln = creal(tmp[0][j]);
        imx = imn = cimag(tmp[0][j]);
        for (i = 1; i < n; i++) {
            tmp_db = creal(tmp[i][j]);
            rlx += (double) (tmp_db > rlx) * (tmp_db - rlx);
            rln += (double) (tmp_db < rln) * (tmp_db - rln);
            tmp_db = cimag(tmp[i][j]);
            imx += (double) (tmp_db > imx) * (tmp_db - imx);
            imn += (double) (tmp_db < imn) * (tmp_db - imn);
        }
        bias[j] = runif(-rlx, -rln) - I * runif(imn, imx);
        for (i = 0; i < n; i++) {
            tmp[i][j] -= bias[j];
            tmp[i][j] = cs_sign(tmp[i][j]);
        }
    }
}

/**
 * @brief Initialize recursively the deep nerual network and compute the nonlinear projection of the subsample
 */
double complex ** net_init(double complex **x, uint32_t n, uint32_t nv_in, uint32_t *nv_ou, dnn *child, uint8_t maxl) {
    uint32_t i, j, k;
    uint32_t const nv_out = *nv_ou;
    double const isrnv = 1.0 / sqrt(nv_in);
    double complex **tmp, **res = NULL;
    uint8_t all_allocated = 1;
    child->n_in = nv_in;
    child->n_out = nv_out;
    child->coef = (double complex *) malloc(nv_in * nv_out * sizeof(double complex));
    tmp = (double complex **) calloc(n, sizeof(double complex *));
    if (tmp) for (i = 0; i < nv_in; i++) {
        tmp[i] = (double complex *) calloc(nv_out, sizeof(double complex));
        all_allocated &= (uint8_t) (tmp[i] != NULL);
    }   
    /* Randomize coefficients */
    if (child->coef && tmp && all_allocated) {
        for (i = 0; i < nv_in; i++) {
            for (j = 0; j < nv_out; j++)
                child->coef[j * nv_in + i] = rnorm(0.0, isrnv) + I * rnorm(0.0, isrnv);
        }
    }
    /* Randomize biases */
    child->bias = (double complex *) malloc(nv_out * sizeof(double complex));
    if (child->bias && tmp && all_allocated) {
        for (i = 0; i < n; i++) {
            for (k = 0; k < nv_out; k++) {
                tmp[i][k] = 0.0;
                for (j = 0; j < nv_in; j++) {
                    tmp[i][k] += child->coef[k * nv_in + j] * x[i][j];
                }
            }
        }
        rnd_bias(child->bias, tmp, n, nv_out);
    }
    if (maxl && tmp && all_allocated) {
        /* Initalize the next layer recursively */
        *nv_ou = 1.0 + rpoisson(0.5 * (double) nv_out);
        child->child = (dnn *) calloc(1, sizeof(dnn));
        if (child->child) {
            res = net_init(tmp, n, nv_out, nv_ou, child->child, maxl - 1);
            if (tmp) for (i = 0; i < nv_in; i++) free(tmp[i]);
            free(tmp);
        }
        return res;
    }
    else {
        child->child = NULL;
        return tmp;
    }
}

/**
 * @brief Swap information at two pointers to `uint32_t`
 * 
 * @param a First pointer to swap
 * @param b Second pointer to swap
 */
static inline void swap(uint32_t *a, uint32_t *b) {
    uint32_t z = *a;
    *a = *b;
    *b = z;
}

/**
 * @brief Subsampling routine
 * 
 * @param nr Number of records in the dataset
 * @param psi Number of subsamples
 */
static inline void sample(uint32_t nr, uint32_t psi) {
    uint32_t i, tmp, sel = 0;
    subs = (uint32_t *) calloc(nr, sizeof(uint32_t));
    if (subs) if (psi == nr) {
        // #pragma omp for simd
        for (i = 0; i < nr; i++)
            subs[i] = 1;
    } 
    else if (psi < nr) {
        for (i = 0; i < psi; i++) subs[i] = 1;
        for (i = 0; i < nr; i++) /** FIXME: change rand because it is not good for openMP */
            swap(&subs[i], &subs[(uint32_t) rand() % nr]);
    }
    else {
        free(subs);
    }
}

/**
 * @brief Initializing projection vector
 * 
 * @param nr Number of records in the dataset
 * @param psi Number of subsamples
 */
static inline void init_proj(uint32_t nr, uint32_t psi) {
    uint32_t i, j;
    if (subs && proj) {
        for (i = j = 0; i < nr && j < psi; i++) {
            proj[j].i = i * (uint32_t) (subs[i] > 0);
            proj[j].v = 0.0;
            j += (uint32_t) (subs[i] > 0);
        }
    }
}

/**
 * @brief Get split based on the most separated values in the projection
 * 
 * @param pstrt Position of subsamples in the sorted vector
 * @param psi Number of subsamples
 * @param szl Pointer to the size of the new left branch
 * @param szr Pointer to the size of the new right branch
 * @return double The splitting value for the projected values
 */
static inline double get_split(uint32_t pstrt, uint32_t psi, uint32_t *szl, uint32_t *szr) {
    uint32_t i, pos;
    double df, mxd = 0.0;
    pos = pstrt;
    for (i = pstrt; i < pstrt + psi - 1; i++) {
        df = proj[i + 1].v - proj[i].v;
        pos += (int32_t) (df > mxd) * (i - pos);
        mxd += (double) (df > mxd) * (df - mxd);
    }
    *szl = pos - pstrt;
    *szr = psi - *szl;
    return runif(proj[pos].v, proj[pos + 1].v);
}

/**
 * @brief Create an isolation tree
 * 
 * @param X Pointer to input data (matrix stored in column-major format)
 * @param pstrt Position of subsamples in the sorted vector
 * @param psi Number of subsamples
 * @param nv Number of variables
 * @param e Current tree height (or depth)
 * @param l Height limit (or depth limit)
 * @return iTrees 
 */
iTrees * iTree(double complex **X, uint32_t pstrt, uint32_t psi, uint32_t nv, uint8_t e, uint8_t const l) {
    uint32_t i, j;
    double p = nan(""), sm = 0.0;
    uint32_t szl = 0, szr = 0;
    iTrees * my_tree = NULL;
    double *w = (double *) malloc((nv << 1) * sizeof(double));
    my_tree = (iTrees *) calloc(1, sizeof(iTrees));
    if (l == 0) my_tree->nv_lat = nv;
    if (my_tree && w) {
        if (e >= l || psi <= 1) {
            my_tree->size = psi;
            my_tree->type = 0;
        }
        else {
            my_tree->level = e;
            my_tree->type = 1;
            /* Draw random projection */
            for (i = 0; i < (nv << 1); i++) {
                w[i] = rnorm(0.0, 1.0);
                sm += w[i] * w[i];
            }
            /* Normalize the projection vector */
            sm = sm > 0.0 ? 1.0 / sqrt(sm) : 1.0;
            for (i = 0; i < (nv << 1); i++) {
                w[i] *= sm;
            }
            my_tree->lincon = w;
            /* Compute the projection vector */
            if (proj) {
                for (i = pstrt; i < pstrt + psi; i++) {
                    proj[i].v = 0.0;
                    for (j = 0; j < nv; j++) {
                        proj[i].v += creal(X[proj[i].i][j]) * w[j << 1];
                        proj[i].v += cimag(X[proj[i].i][j]) * w[1 | (j << 1)];
                    }
                }
                /* Sort the projection subvector */
                qsort(&proj[pstrt], psi, sizeof(vector), cmp_vec);
                /* Get threshold and update size left and right */
                p = get_split(pstrt, psi, &szl, &szr);
                /* Create next set of trees */
                e++;
                my_tree->left = iTree(X, pstrt, szl, nv, e, l);
                my_tree->right = iTree(X, pstrt + szl, szr, nv, e, l);
            }
            my_tree->threshold = p;
        }
    }
    return my_tree;
}

static inline void populate_csubx(double complex **csubx, uint32_t psi, uint32_t nv, uint32_t nr, double *X) {
    uint32_t i, j;
    for (i = 0; i < psi; i++) {
        for (j = 0; j < nv; j++) {
            csubx[i][j] = X[nr * j + proj[i].i] + I * 0.0;
        }
        proj[i].i = i;
    }
}

/**
 * @brief Create a Deep Isolation Forest (DIF)
 * 
 * @param X Pointer to input data (matrix stored in column-major format) 
 * @param dimX Pointer to number of rows and columns of `X` FIXME: we may need thd single values separated
 * @param nt Pointer to number of trees
 * @param nss Pointer to subsampling size
 * @return iTrees* Pointer to a Generalized Isolation Forest (a set of iTrees)
 */
iModels deep_iForest(double *X, int *dimX, int *nt, int *nss) {
    iModels iModel;
    double complex **csubx, **sub_net_out;
    uint8_t all_allocated = 1;
    uint32_t psi = *(uint32_t *) nss;
    uint32_t nr = (uint32_t) dimX[0];
    uint32_t nv = (uint32_t) dimX[1];
    uint32_t nlatent;
    uint32_t i, j, t = *(uint32_t *) nt;
    uint8_t const l = (uint8_t) (log2((double) (psi + (psi < 1))) + 0.5);
    psi = (uint32_t) (psi > nr) * nr + (uint32_t) (psi <= nr) * psi;

    if (l > 30) {
        iModel.isoForest = NULL;
        iModel.isoNet = NULL;
        return iModel;
    }
    srand(time(NULL));
    iModel.isoForest = (iTrees **) calloc(t, sizeof(iTrees *));
    iModel.isoNet = (dnn **) calloc(t, sizeof(dnn *));
    proj = (vector *) malloc(psi * sizeof(vector));
    csubx = (double complex **) calloc(psi, sizeof(double complex *));
    if (csubx) for (i = 0; i < psi; i++) {
        csubx[i] = (double complex *) calloc(nv, sizeof(double complex));
        all_allocated &= (uint8_t) (csubx[i] != NULL);
    }
    // #pragma omp parallel for
    if (iModel.isoForest && iModel.isoNet && proj && csubx && all_allocated) {
        for (i = 0; i < t; i++) {
            sample(nr, psi);
            if (subs) {
                init_proj(nr, psi);
                /** Complexify subsample of X */
                populate_csubx(csubx, psi, nv, nr, X);
                /* Initialize the network and valuate the network */
                nlatent = 1 + rpoisson(1.0 + 0.5 * (double) nv);
                sub_net_out = net_init(csubx, psi, nv, &nlatent, \
                        iModel.isoNet[i], 2 + rpoisson(1.0));
                if (sub_net_out) {
                    iModel.isoForest[i] = iTree(sub_net_out, 0, psi, nlatent, 0, l);
                    for (j = 0; j < nlatent; i++) free(sub_net_out[j]);
                    free(sub_net_out);
                }
            }
            free(subs);
        }
    }
    free(proj);
    if (csubx) for (i = 0; i < psi; i++) free(csubx[i]);
    free(csubx);
    return iModel;
}

/**
 * @brief Compute the isolation score (or path length)
 * 
 * @param x Pointer to an input vector
 * @param nv Number of variables (i.e. length of `x`) 
 * @param tree Pointer to a tree in the forest
 * @param e current path length 
 * @return double Isolation score
 */
double path_length(double complex *x, uint32_t nv, iTrees *tree, uint8_t e) {
    double res = (double) e;
    double prjx = 0.0;
    uint32_t i;
    if (tree->type == 0) {
        if (tree->size > 1) res += cfun(tree->size);
        return res;
    }
    else {
        for (i = 0; i < nv; i++) {
            prjx += creal(x[i]) * tree->lincon[i << 1];
            prjx += cimag(x[i]) * tree->lincon[1 | (i << 1)];
        }
        e++;
        return path_length(x, nv, prjx <= tree->threshold ? tree->left : tree->right, e);
    }
}

/**
 * @brief Compute the anomaly score of a generalized isolation forest
 * 
 * @param x Pointer to an input vector
 * @param Forest Pointer to a foreset (i.e., double pointer to trees)
 * @param t Pointer to number of trees in the forest
 * @param psi Pointer to number of subsamples used to construct the trees in the forestuint32_t t, 
 * @return double 
 */
double fuzzy_anomaly_score(double complex *x, iModels *model, int *t, int *psi) {
    double avglen = 1.0;
    double complex *nlp; /* Nonlinear projections */
    uint32_t i;
    uint32_t it = (uint32_t) *t;
    double const nrmc = -1.0 / cfun((uint32_t) *psi);
    for (i = 0; i < it; i++) {
        nlp = net_eval(x, model->isoNet[i], 0);
        if (nlp) {
            avglen += log1p(- pow(2.0, \
            path_length(nlp, (uint32_t) model->isoForest[i]->nv_lat, model->isoForest[i], 0) * nrmc));
        }
        free(nlp);
    }
    avglen /= (double) it;
    return 1.0 - exp(avglen);
}

/**
 * @brief Free the memory allocated for a tree
 * 
 * @param tree Pointer to an isolation tree structure 
 */
void free_tree(iTrees *tree) {
    free(tree->lincon);
    if (tree->type) {
        free_tree(tree->left);
        free_tree(tree->right);
    }
    free(tree);
}

/**
 * @brief Free the memory allocated for a neural network
 * 
 * @param net Pointer to a neural network structure
 */
void free_dnn(dnn *net) {
    if (net->child) free_dnn(net->child);
	free(net->coef);
	free(net->bias);
	free(net);
}

/**
 * @brief Free the memory allocated for the forest
 * 
 * @param Forest Pointer to pointer of the allocated tress
 * @param t Number of trees in the forest
 */
static inline void free_model(iModels *m, int *t) {
    uint32_t i;
    for (i = 0; i < (uint32_t) *t; i++) {
        free_dnn(m->isoNet[i]);
        free_tree(m->isoForest[i]);
    }
    free(m->isoNet);
    free(m->isoForest);
}

/**
 * @brief Deep Isolation Forest
 *
 * @param res Pointer to an empty vector where to store the anomaly scores 
 * @param dta Pointer to the values of a dataset
 * @param dimD Pointer to the dimensions of a dataset
 * @param nt Pointer to the number of ensemble models
 * @param nss Pointer to the number of subsamples to draw from the dataset without replacement
 */
void dif(double *res, double *dta, int *dimD, int *nt, int *nss) {
    uint32_t i, j;
    iModels model;
    double complex *cat;
    
    if (*nss <= 0) return;
    if (*nt <= 0) return;
    if (dimD[0] <= 0 || dimD[1] <= 0) return;

    H = (double *) malloc(dimD[0] * sizeof(double));
    cat = (double complex *) malloc(dimD[1] * sizeof(double complex));
    if (H && cat) {
        H[0] = 1.0;
        for (i = 1; i < (uint32_t) *nss; i++) 
            H[i] = H[i - 1] + 1.0 / (1.0 + (double) i);
        model = deep_iForest(dta, dimD, nt, nss);
        if (model.isoForest && model.isoNet) {
            for (i = 0; i < (uint32_t) dimD[0]; i++) {
                for (j = 0; j < (uint32_t) dimD[1]; j++)
                    cat[j] = dta[dimD[0] * j + i] + I * 0.0;
                res[i] = fuzzy_anomaly_score(cat, &model, nt, nss);
            }
        }
        free_model(&model, nt);
    }
    free(H);
    free(cat);
}

/*
data(iris)
dyn.load("test.so")
dta <- iris[, 1L:4L]
anom <- .C("dif", res = double(nrow(dta)), as.matrix(dta), dim(dta), 10L, 32L)$res
graphics.off()
hist(anom)
X11()
pairs(dta, col = iris$Species, pch = c(".", "+")[1+(anom > 0.5)], cex = 2)
mean(anom > 0.5)
*/
