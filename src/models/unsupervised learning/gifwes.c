#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include <omp.h>

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
    double threshold;
} iTrees;

double *H;
uint32_t *subs;
double *proj;
// #pragma omp threadprivate(subs, proj)

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
 * The function rnorm() is a C function that generates a random number from a normal distribution with
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
    proj = (double *) malloc(nr * sizeof(double));
    if (subs) if (psi == nr) {
        // #pragma omp for simd
        for (i = 0; i < nr; i++)
            subs[i] = 1;
    } 
    else if (psi < nr) {
        for (i = 0; i < psi; i++) subs[i] = 1;
        for (i = 0; i < nr; i++) /** FIXME: change algorithm because rand is not good for openMP */
            swap(&subs[i], &subs[(uint32_t) rand() % nr]);
    }
    else {
        free(subs);
    }
}

/**
 * @brief Create an isolation tree
 * 
 * @param X Pointer to input data (matrix stored in column-major format) 
 * @param psi Number of subsamples
 * @param nr Number of records
 * @param nv Number of variables
 * @param rl Right (= 1) or left (= 0) indicator 
 * @param e Current tree height (or depth)
 * @param l Height limit (or depth limit)
 * @return iTrees 
 */
iTrees * iTree(double *X, uint32_t psi, uint32_t nr, uint32_t nv, uint32_t rl, uint8_t e, uint8_t const l) {
    uint32_t i, j;
    double mn, mx, p = nan(""), sm = 0.0;
    uint32_t szl = 0, szr = 0;
    iTrees * my_tree = NULL;
    double *w = (double *) malloc(nv * sizeof(double));
    my_tree = (iTrees *) calloc(1, sizeof(iTrees));
    if (my_tree && w) {
        if (e >= l || psi <= 1) {
            my_tree->size = psi;
            my_tree->type = 0;
        }
        else {
            my_tree->level = e;
            my_tree->type = 1;
            /* Draw random projection */
            for (i = 0; i < nv; i++) {
                w[i] = rnorm(0.0, 1.0);
                sm += w[i] * w[i];
            }
            /* Normalize the projection vector */
            sm = sm > 0.0 ? 1.0 / sqrt(sm) : 1.0;
            for (i = 0; i < nv; i++) {
                w[i] *= sm;
            }
            my_tree->lincon = w;
            /* Compute the projection vector */
            if (subs && proj) {
                mn = (double) INFINITY;
                mx = -mn; 
                for (i = 0; i < nr; i++) {
                    /* First bit (`& 1`) says if the data point was or not in the subsample of the root node */
                    /* Other bits (`>> 1`) say the partition number at the current node */
                    if ((subs[i] & 1) && ((subs[i] >> 1) == rl)) {
                        proj[i] = 0.0;
                        for (j = 0; j < nv; j++) {
                            proj[i] += X[nr * j + i] * w[j];
                        }
                        /* Compute min and max of the projection */
                        if (mn > proj[i]) mn = proj[i];
                        if (mx < proj[i]) mx = proj[i];
                    }
                }
                /* Draw the threshold */
                p = runif(mn, mx);
                /* Split the subsample in input */
                for (i = 0; i < nr; i++) {
                    if ((subs[i] & 1) && ((subs[i] >> 1) == rl)) {
                        j = (uint32_t) (proj[i] > p);
                        szl += 1 - j;
                        szr += j;
                        subs[i] = (rl << 2) | (j << 1) | (subs[i] & 1);
                    }
                }
                /* Create next set of trees */
                e++;
                rl <<= 1;
                my_tree->left = iTree(X, szl, nr, nv, rl, e, l);
                my_tree->right = iTree(X, szr, nr, nv, rl + 1, e, l);
            }
            my_tree->threshold = p;
        }
    }
    return my_tree;
}


/**
 * @brief Create a Generalized Isolation Forest (GIF)
 * 
 * @param X Pointer to input data (matrix stored in column-major format) 
 * @param dimX Pointer to number of rows and columns of `X`
 * @param nt Pointer to number of trees
 * @param nss Pointer to subsampling size
 * @return iTrees* Pointer to a Generalized Isolation Forest (a set of iTrees)
 */
iTrees ** iForest(double *X, int *dimX, int *nt, int *nss) {
    iTrees **Forest = NULL;
    uint32_t psi = *(uint32_t *) nss;
    uint32_t nr = (uint32_t) dimX[0];
    uint32_t nv = (uint32_t) dimX[1];
    uint32_t i, t = *(uint32_t *) nt;
    uint8_t const l = (uint8_t) (log2((double) (psi + (psi < 1))) + 0.5);
    psi = (uint32_t) (psi > nr) * nr + (uint32_t) (psi <= nr) * psi;

    if (l > 30) return NULL;
    srand(time(NULL));
    Forest = (iTrees **) calloc(t, sizeof(iTrees *));
    // #pragma omp parallel for
    if (Forest) for (i = 0; i < t; i++) {
            sample(nr, psi);
            if (subs && proj) Forest[i] = iTree(X, psi, nr, nv, 0, 0, l);
            free(subs);
            free(proj);
    }
    return Forest;
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
double path_length(double *x, uint32_t nv, iTrees *tree, uint8_t e) {
    double res = (double) e;
    double prjx = 0.0;
    uint32_t i;
    if (tree->type == 0) {
        if (tree->size > 1) res += cfun(tree->size);
        return res;
    }
    else {
        for (i = 0; i < nv; i++) {
            prjx += x[i] * tree->lincon[i];
        }
        e++;
        return path_length(x, nv, prjx <= tree->threshold ? tree->left : tree->right, e);
    }
}

/**
 * @brief Compute the anomaly score of a generalized isolation forest
 * 
 * @param x Pointer to an input vector
 * @param nv Number of variables
 * @param Forest Pointer to a foreset (i.e., double pointer to trees)
 * @param t Pointer to number of trees in the forest
 * @param psi Pointer to number of subsamples used to construct the trees in the forestuint32_t t, 
 * @return double 
 */
double enhanced_anomaly_score(double *x, int nv, iTrees **Forest, int *t, int *psi) {
    double avglen = 0.0;
    uint32_t i;
    uint32_t it = (uint32_t) *t;
    double const ic = -1.0 / cfun((uint32_t) *psi);
    for (i = 0; i < it; i++) {
        avglen += pow(2.0, ic * path_length(x, (uint32_t) nv, Forest[i], 0));
    }
    avglen /= (double) it;
    return avglen;
}

/**
 * @brief Free the memory allocated for a tree
 * 
 * @param tree 
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
 * @brief Free the memory allocated for the forest
 * 
 * @param Forest Pointer to pointer of the allocated tress
 * @param t Number of trees in the forest
 */
static inline void free_forest(iTrees **Forest, int *t) {
    uint32_t i;
    for (i = 0; i < (uint32_t) *t; i++) {
        free_tree(Forest[i]);
    }
    free(Forest);
}

void gif(double *res, double *dta, int *dimD, int *nt, int *nss) {
    uint32_t i, j;
    iTrees **forest;
    double *dat;
    
    if (*nss <= 0) return;
    if (*nt <= 0) return;
    if (dimD[0] <= 0 || dimD[1] <= 0) return;

    H = (double *) malloc(dimD[0] * sizeof(double));
    dat = (double *) malloc(dimD[1] * sizeof(double));
    if (H && dat) {
        H[0] = 1.0;
        for (i = 1; i < (uint32_t) *nss; i++) 
            H[i] = H[i - 1] + 1.0 / (1.0 + (double) i);
        forest = iForest(dta, dimD, nt, nss);
        if (forest) {
            for (i = 0; i < (uint32_t) dimD[0]; i++) {
                for (j = 0; j < (uint32_t) dimD[1]; j++)
                    dat[j] = dta[dimD[0] * j + i];
                res[i] = enhanced_anomaly_score(dat, dimD[1], forest, nt, nss);
            }
        }
        free_forest(forest, nt);
    }
    free(H);
    free(dat);
}

/*
data(iris)
dyn.load("test.so")
dta <- iris[, 1L:4L]
anom <- .C("gif", res = double(nrow(dta)), as.matrix(scale(dta)), dim(dta), 1000L, 108L)$res
hist(anom)
pairs(dta, col = iris$Species, pch = c(1, 19)[1+(anom > 0.7)], cex = 2)
mean(anom > 0.7)
*/
