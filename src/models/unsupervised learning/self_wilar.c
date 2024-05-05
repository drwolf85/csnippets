#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <omp.h>

#define MAX_REG_ITER 1000

typedef struct deep_neural_network {
	struct deep_neural_network *child;
	uint32_t n_in;
	uint32_t n_out;
	double complex *coef;
} dnn;

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

typedef struct vec_proj {
	double v;
	uint32_t i;
} vector;

double *H;
uint32_t *subs;
vector *proj;
double complex *x;
double complex *y;

/**
@brief Residuals of a weighted linear model (based on OLS)
@param res empty vector to store the residuals (in output)
@param py response vector (example data for model output)
@param w vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
static inline void wlm_resid(double *res, double *py, double *w, double *pdta, int *dim) {
    int i, j, k;
    double itmp, tmp, v;
    double *q, *yy, *dta, *iw, *vec;

    q = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    dta = (double *) malloc(dim[0] * dim[1] * sizeof(double));
    yy = (double *) malloc(dim[0] * sizeof(double));
    iw = (double *) malloc(dim[0] * sizeof(double));
    vec = (double *) malloc(dim[1] * sizeof(double));
    memset(res, 0, dim[0] * sizeof(double));
    if (q && yy && dta && iw && vec) {
        /* Adjust data for the weights */
        for (i = 0; i < dim[0]; i++) {
            iw[i] = sqrt(w[i]);
            yy[i] = py[i] * iw[i];
            #pragma omp for simd
            for (j = 0; j < dim[1]; j++) {
                dta[*dim * j + i] = pdta[*dim * j + i] * iw[i];
            }
            iw[i] = 1.0 / iw[i];
        }
        /* Computing matrix Q */
        for (i = 0; i < dim[1]; i++) {
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] = dta[*dim * i + j];
            for (k = 0; k < i; k++) {
                tmp = 0.0;
                v = 0.0;
                for (j = 0; j < dim[0]; j++) {
                    tmp += q[*dim * k + j] * dta[*dim * i + j];
                    v += q[*dim * k + j] * q[*dim * k + j];
                }
                tmp /= v;
                #pragma omp for simd
                for (j = 0; j < dim[0]; j++) 
                    q[*dim * i + j] -= tmp * q[*dim * k + j];
            }
            /* Normalization of the column vector */
            tmp = 0.0;
            for (j = 0; j < dim[0]; j++)
                tmp += q[*dim * i + j] * q[*dim * i + j];
            itmp = 1.0 / sqrt(tmp);
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++)
                q[*dim * i + j] *= itmp;
        }
        /* Computing least square residuals (Q^t W^{0.5} y)*/
        for (k = 0; k < dim[1]; k++) {
            tmp = 0.0;
            #pragma omp for simd
            for (j = 0; j < dim[0]; j++) {
                tmp += q[*dim * k + j] * yy[j];
            }
            vec[k] = tmp;
        }
        /* Computing least square residuals y - W^{-0.5} (QQ^t) W^{0.5} y */
        for (j = 0; j < dim[0]; j++) {
            tmp = 0.0;
            for (k = 0; k < dim[1]; k++) {
                tmp += q[*dim * k + j] * vec[k];
            }
            res[j] = yy[j] - tmp;
            res[j] *= iw[j];
        }
    }
    free(q);
    free(yy);
    free(iw);
    free(vec);
    free(dta);
}

/**
@brief Weighted iterated least absolute residuals
@param res empty vector to store the residulas
@param py response vector (example data for model output)
@param pw vector of weights
@param pdta matrix of data (column-major format)
@param dim vector of dimension of `dta` matrix
*/
static inline void wilar_resid(double *res, double *py, double *pw, double *pdta, int *dim) {
    int i, j, k = 0;
    double *w;

    w = (double *) malloc(dim[0] * sizeof(double));
    /* Get OLS residuals for a weighted linear model */
    wlm_resid(res, py, pw, pdta, dim);
    if (w) do { /* Get WOLS residuals (iteratively) */
        #pragma omp for simd
        for (i = 0; i < dim[0]; i++) { /* Compute the weights */
            w[i] = pw[i] / fabs(res[i]);
            w[i] = isfinite(w[i]) ? w[i] : pw[i];
        }
        wlm_resid(res, py, w, pdta, dim);
        k++;
    }
    while (k < MAX_REG_ITER);
    free(w);
}


static inline double cfun(uint32_t n) {
    return 2.0 * (H[n - 2] - (double) (n - 1) / (double) n);
}

static inline double complex cs_sign(double complex x) {
    return x / (1.0 + cabs(x));
}

static inline double runif(double a, double b) {
    unsigned long u, m = ~(1 << 31);
        u = rand() & m;
        return ldexp((double) u, -31) * (b - a) + a;
}

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

static inline double dpoisson(uint32_t x, double lambda) {
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

static inline uint32_t qpoisson(double p, double lambda) {
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

static inline uint32_t rpoisson(double lambda) {
    uint32_t u, m, z = 0;
    if (lambda >= 0.0) {
        u = rand();
        m = ~(1 << 31);
        u &= m;
        z = qpoisson(ldexp((double) u, -31), lambda);
    }
    return z;
}

static inline void swap(uint32_t *a, uint32_t *b) {
    uint32_t z = *a;
    *a = *b;
    *b = z;
}

static inline void sample(uint32_t nr, uint32_t psi) {
    uint32_t i;
    subs = (uint32_t *) calloc(nr, sizeof(uint32_t));
    if (subs) { 
        if (psi == nr) {
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
}

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


dnn * net_init(uint32_t n_in, uint8_t n_ly) {
    uint32_t i, j, n_out;
    double const isqrtn = 1.0 / sqrt(n_in);
    dnn *layer = (dnn *) calloc(1, sizeof(dnn));
    if (layer) {
        layer->n_in = n_in;
        n_out = 1 + rpoisson(0.5 * (double) n_in);
        layer->n_out = n_out;
        layer->coef = (double complex *) malloc(n_in * n_out * sizeof(double complex));
        if (layer->coef) {
            for (i = 0; i < n_in; i++) {
                for (j = 0; j < n_out; j++) {
                    layer->coef[j * n_in + i] = rnorm(0.0, isqrtn) + I * rnorm(0.0, isqrtn);;
                }
            }
        }
        layer->child = n_ly ? net_init(layer->n_out, n_ly - 1) : NULL;
    }
    return layer;
}

void net_free(dnn *net) {
    if (net->child) net_free(net->child);
	free(net->coef);
	free(net);
}

double complex ** out_vec_alloc(dnn *layer, uint8_t n_ly, uint32_t nr) {
    dnn *next = layer;
    uint8_t i = 0;
    double complex **vecs = (double complex **) calloc(n_ly + 1, sizeof(double complex *));
    if (vecs && layer) {
        do {
            vecs[i] = (double complex *) calloc(next->n_out * nr, sizeof(double complex));
            next = next->child;
            i++;
        } while(next);
    }
    return vecs;
}

void out_vec_free(double complex **vecs, uint8_t nly) {
    uint8_t i = 0;
    if (vecs) {
        for (; i <= nly; i++) free(vecs[i]);
        free(vecs);
    }
}

int cmp_vec(void const *aa, void const *bb) {
    vector a = *(vector *) aa;
    vector b = *(vector *) bb;
    return (int) (a.v > b.v) - (int) (a.v < b.v);
}

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

iTrees * iTree(double complex *X, uint32_t pstrt, uint32_t psi, uint32_t nv, uint8_t e, uint8_t const l) {
    uint32_t i, j;
    double p = nan(""), sm = 0.0;
    uint32_t szl = 0, szr = 0;
    iTrees *my_tree = NULL;
    double *w = NULL;
    my_tree = (iTrees *) calloc(1, sizeof(iTrees));
    my_tree->lincon = (double *) calloc((nv << 1), sizeof(double));
    w = my_tree->lincon;
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
            /* Compute the projection vector */
            if (proj) {
                for (i = pstrt; i < pstrt + psi; i++) {
                    proj[i].v = 0.0;
                    for (j = 0; j < nv; j++) {
                        proj[i].v += creal(X[proj[i].i * nv + j]) * w[j << 1];
                        proj[i].v += cimag(X[proj[i].i * nv + j]) * w[1 | (j << 1)];
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

void free_tree(iTrees *tree) {
    free(tree->lincon);
    if (tree->type) {
        free_tree(tree->left);
        free_tree(tree->right);
    }
    free(tree);
}

void iso_model(uint32_t t, double *res, double complex *dta_row_maj, uint32_t nr, uint32_t nv, uint32_t psi) {
    uint32_t i, j, k = 0, l = 0;
    uint8_t lev = (uint8_t) (log2((double) (psi + (psi < 1))) + 0.5);
    uint8_t const nly = (uint8_t) (2 + rpoisson(1.0));
    dnn *next;
    dnn *net_mod = net_init(nv, nly);
    double complex **vecs = out_vec_alloc(net_mod, nly, nr);
    double complex bias;
    double rln, imn, rlx, imx, tmp_db;
    double val, nrmc;
    double const it = 1.0 / (double)(t + 1);
    iTrees *mytree = NULL;
    psi = (uint32_t) (psi > nr) * nr + (uint32_t) (psi <= nr) * psi;
    nrmc = -1.0 / cfun((uint32_t) psi);
    lev = lev > 30 ? 30 : lev;
    if (net_mod && vecs && proj) {
        next = net_mod;
        for (i = 0; i < nr; i++) {
            for (j = 0; j < next->n_out; j++) {
                for (k = 0; k < next->n_in; k++) {
                    vecs[l][next->n_out * i + j] += \
                    dta_row_maj[next->n_in * i + k] * \
                    next->coef[next->n_in * j + k];
                }
            }
        }
        for (j = 0; j < next->n_out; j++) {
            rlx = rln = creal(vecs[l][j]);
            imx = imn = cimag(vecs[l][j]);
            for (i = 1; i < nr; i++) {  
                tmp_db = creal(vecs[l][next->n_out * i + j]);
                rlx += (double) (tmp_db > rlx) * (tmp_db - rlx);
                rln += (double) (tmp_db < rln) * (tmp_db - rln);
                tmp_db = cimag(vecs[l][next->n_out * i + j]);
                imx += (double) (tmp_db > imx) * (tmp_db - imx);
                imn += (double) (tmp_db < imn) * (tmp_db - imn);
            }
            bias = runif(-rlx, -rln) - I * runif(imn, imx);
            for (i = 0; i < nr; i++) {
                vecs[l][next->n_out * i + j] = \
                cs_sign(vecs[l][next->n_out * i + j] - bias);
            }
        }
        k = next->n_out;
        next = next->child;
        l++;
        while(next) {
            for (i = 0; i < nr; i++) {
                for (j = 0; j < next->n_out; j++) {
                    for (k = 0; k < next->n_in; k++) {
                        vecs[l][next->n_out * i + j] += \
                        vecs[l-1][next->n_in * i + k] * \
                        next->coef[next->n_in * j + k];
                    }
                }
            }
            for (j = 0; j < next->n_out; j++) {
                rlx = rln = creal(vecs[l][j]);
                imx = imn = cimag(vecs[l][j]);
                for (i = 1; i < nr; i++) {  
                    tmp_db = creal(vecs[l][next->n_out * i + j]);
                    rlx += (double) (tmp_db > rlx) * (tmp_db - rlx);
                    rln += (double) (tmp_db < rln) * (tmp_db - rln);
                    tmp_db = cimag(vecs[l][next->n_out * i + j]);
                    imx += (double) (tmp_db > imx) * (tmp_db - imx);
                    imn += (double) (tmp_db < imn) * (tmp_db - imn);
                }
                bias = runif(-rlx, -rln) - I * runif(imn, imx);
                for (i = 0; i < nr; i++) {
                    vecs[l][next->n_out * i + j] = \
                    cs_sign(vecs[l][next->n_out * i + j] - bias);
                }
            }
            k = next->n_out;
            next = next->child;
            l++;
        }
    }
    net_free(net_mod);
    proj = (vector *) malloc(psi * sizeof(vector));
    sample(nr, psi);
    init_proj(nr, psi);
    if (proj && subs) {
        mytree = iTree(vecs[nly], 0, psi, k, 0, lev);
        if (mytree) {
            for (i = 0; i < nr; i++) {
                val = path_length(&vecs[nly][k * i], k, mytree, 0) * nrmc;
                res[i] *= (double) t * it;
                res[i] += it * log1p(- pow(2.0, val)); /* * path_length... */
            }
        }
    }
    free(subs);
    free(proj);
    out_vec_free(vecs, nly);    
    free_tree(mytree);
}

void dif(double *res, double *dta, int *dimD, int *nt, int *nss) {
    uint32_t i, t;
    srand(time(NULL));
    H = (double *) malloc(dimD[0] * sizeof(double));
    double complex *dta_row_major = (double complex *) \
        malloc(dimD[0] * dimD[1] * sizeof(double complex));
    if (dta_row_major && H) {
        H[0] = 1.0;
        for (i = 1; i < (uint32_t) *nss; i++) 
            H[i] = H[i - 1] + 1.0 / (1.0 + (double) i);
        for (i = 0; i < dimD[0]; i++) {
            for (t = 0; t < dimD[1]; t++) {
                dta_row_major[dimD[1] * i + t] = \
                    dta[dimD[0] * t + i] + I * 0.0;
            }
        }
        for (t = 0; t < (uint32_t) *nt; t++) {
            iso_model(t, res, dta_row_major, \
                (uint32_t) *dimD, (uint32_t) dimD[1], \
                (uint32_t) *nss);
        }
        for (i = 0; i < (uint32_t) *dimD; i++) {
            res[i] = 1.0 - exp(res[i]);
        }
    }
    free(dta_row_major);
    free(H);
}

/**
 * @brief Robust linear self-smoothing matrix
 * 
 * @param X Pointer to a matrix
 * @param dimX Pointer to number of rows and columns
 * @param w Pointer to a vector of weights
 */
void rlssm(double *X, int *dimX, double *w) {
    uint32_t i, j, s;
    uint32_t n = (uint32_t) *dimX;
    uint32_t p = (uint32_t) dimX[1];
    double *finX = (double *) malloc(n * p * sizeof(double)); 
    double *pdta = (double *) malloc(n * p * sizeof(double));
    if (pdta && finX && p > 1 && n >= p) {
        for (i = 0; i < n; i++) pdta[i] = 1.0;
        for (j = 1; j < p; j++)
            for (i = 0; i < n; i++) 
                pdta[j * n + i] = X[j * n + i];
        wilar_resid(finX, X, w, pdta, dimX);
        for (s = 1; s < p; s++) {
            for (j = 0; j < s; j++) 
                for (i = 0; i < n; i++) 
                    pdta[(j + 1) * n + i] = X[j * n + i]; /** FIMXE: */
            wilar_resid(&finX[s * n], &X[s * n], w, pdta, dimX);
        }
        for (i = 0; i < n * p; i++) X[i] -= finX[i];
    }
    free(pdta);
    free(finX);
}

/**
 * @brief Comparison function for double values
 *
 * @param aa Pointer to double
 * @param bb Pointer to double
 * @return int
 */
int cmp_double(void const *aa, void const *bb) {
    double a = *(double *)aa;
    double b = *(double *)bb;
    if (isnan(a)) return 1;
    if (isnan(b)) return -1;
    return (int) (a >= b) * 2 - 1;
}

/**
 * @brief Autoscaling function
 * 
 * @param X Pointer to a matrix of data (column-major format)
 * @param dimX Pointer to a vector with the number of rows and columns of `X`
 * @param center Boolean value, it indicates if centering is required
 * @param scale Boolean value, it indicates if scaling is required
 */
void autoscale(double *X, int *dimX, bool center, bool scale) {
    size_t i, ii, j, pos, nn;
    size_t n = (size_t) *dimX;
    size_t p = (size_t) dimX[1];
    double m, s, tmp;
    double *da;
    da = (double *) malloc(n * sizeof(double));
    if (da) {
        for (j = 0; j < p; j++) {
            nn = 0;
            #pragma omp simd
            for (i = 0; i < n; i++) da[i] = X[j * n + i];
            qsort(da, n, sizeof(double), cmp_double);
            for (i = 0; i < n; i++) nn += (size_t) isnan(da[i]);
            nn = n - nn;
            if (nn == 0) continue;
            i = nn >> 1;
            pos = i - (size_t)!(nn & 1 || i == 0);
            m = da[i] * 0.5 + da[pos] * 0.5;
            ii = i >> 1;
            pos = i + ii - (size_t)!(i & 1 || i + ii == 0);
            s = da[i + ii] * 0.5 + da[pos] * 0.5;
            pos = ii - (size_t)!(i & 1 || ii == 0);
            s -= da[ii] * 0.5 + da[pos] * 0.5;
            s *= 2.833030804858356788145;
            if (s <= 0.0) {
                s = 0.0;
                ii = 0;
                for (i = 0; i < nn; i++) {
                    tmp = fabs(da[i] - m);
                    ii += (size_t) (tmp > 0.0);
                    s += tmp;
                }
                s /= ii > 0 ? ii : 1.0;
            }
            s = s > 0.0 ? 1.0 / s : 1.0;
            for (i = 0; i < n; i++) {
                X[j * n + i] -= m * (double) center;
                X[j * n + i] *= s * (double) scale + (double) !scale;
            }
        }
    }
    free(da);
}

#ifdef DEBUG
#include "../../.data/iris.h"
int main() {
    int j, i = 0, ntrees = 1000, nss = 32;
    double w[N] = {0};
    double dta[N * P];
    int dim[2] = {N, P};
    for (i = 0; i < N * P; i++) dta[i] = log(x_iris[i]);
    autoscale(dta, dim, true, true);
    dif(w, dta, dim, &ntrees, &nss);
    for (i = 0; i < N; i++) w[i] = 1.0 / w[i];
    for (i = 0; i < N * P; i++) dta[i] = log(x_iris[i]);
    rlssm(dta, dim, w);
    for (i = 0; i < N * P; i++) dta[i] = exp(dta[i]);
    for (i = 0; i < N; i++) {
        for (j = 0; j < P; j++)
            printf("%1.2f ", x_iris[i + N * j]);
        printf("\t");
        for (j = 0; j < P; j++)
            printf("%1.2f ", dta[i + N * j]);
        printf("\t");
        for (j = 0; j < P; j++)
            printf("%s%1.2f ", x_iris[i + N * j] >= dta[i + N * j] ? " " : "", x_iris[i + N * j] - dta[i + N * j]);            
        printf("\n"); 
    }
    return 0;
}
#endif

