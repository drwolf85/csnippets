#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MY_EPS_RATIO_TOLL 1e-20

/** 
 * @brief The function initializes the coefficient of the polynom
 *
 * @param p Order of the polynom
 * @param rnd_coef Pointer to a function that generates random coefficients
 *
 * @return A pointer to an array of random polynomial coefficients
 */
double * init_poly(int p, double (*rnd_coef)(void)) {
    int i = 0;
    double *polynom = (double *) malloc((p + 1) * sizeof(double));
    if (polynom) {
        for (; i <= p; i++) {
            polynom[i] = rnd_coef();
        } 
    }
    return polynom;
}

/** 
 * @brief The function frees the memory used to store the polynomial coefficient
 *
 * @param polynom A pointer to an allocated array of random polynomial coefficients
 */
void free_poly(double *polynom) {
    free(polynom);
}

/** 
 * @brief The function evaluates the random polynom at a generic value x
 * 
 * @param x Generic value (double precision floating point) where the polynom is evaluated
 * @param polynom A pointer to an allocated array of random polynomial coefficients
 * @param p Number of polynomial coefficients randomly generated
 *
 * @return Value of the random polynom computed in `x`
 */
double eval_poly(double x, double *polynom, int p) {
    int i = 1;
    double tmp = 1.0;
    double res = *polynom;
    for (; i <= p; i++) {
        tmp *= x / (double) i;
        res += polynom[i] * tmp;
    }
    return res;
}

/** 
 * @brief The function evaluates the random polynom at a generic value x and applies hard clipping
 * 
 * @param x Generic value (double precision floating point) where the polynom is evaluated
 * @param polynom A pointer to an allocated array of random polynomial coefficients
 * @param p Number of polynomial coefficients randomly generated
 * @param clipping Value (double precision floating point) used for clipping the polynomial output
 *
 * @return Value of the random polynom computed in `x`
 */
double eval_poly_n_hard_clip(double x, double *polynom, int p, double clipping) {
    int i = 1;
    double tmp = 1.0;
    double res = *polynom;
    for (; i <= p; i++) {
        tmp *= x / (double) i;
        res += polynom[i] * tmp;
    }
    clipping = fabs(clipping);
    res = (double) (res >= clipping) * clipping - \
          (double) (res <= -clipping) * clipping + \
          (double) (res > -clipping && res < clipping) * res;
    return res;
}

/** 
 * @brief The function evaluates the random polynom at a generic value x and applies soft clipping
 * 
 * @param x Generic value (double precision floating point) where the polynom is evaluated
 * @param polynom A pointer to an allocated array of random polynomial coefficients
 * @param p Number of polynomial coefficients randomly generated
 * @param clipping Value (double precision floating point) used for soft clipping the polynomial output
 *
 * @return Value of the random polynom computed in `x`
 */
double eval_poly_n_soft_clip(double x, double *polynom, int p, double clipping) {
    int i = 1;
    double tmp = 1.0;
    double res = *polynom;
    for (; i <= p; i++) {
        tmp *= x / (double) i;
        res += polynom[i] * tmp;
    }
    clipping = MY_EPS_RATIO_TOLL + fabs(clipping);
    res = res * clipping / (clipping + fabs(res));
    return res;
}

#ifdef DEBUG

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

static inline double rbern(double prob) {
    unsigned long u, m;
    double z;
    u = rand();
    m = ~(1 << 31);
    u &= m;
    z = (double) (ldexp((double) u, -31) < prob);        
    return z;
}

double norm_coef(void) {
    return rnorm(0.0, 1.0);
}

double spike_n_slab_coef(void) {
    return rnorm(0.0, 1.0) * rbern(0.75);
}

#define N_SPLITS 20
#define N_COEF 13
#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0
#define MY_CLIP 5.678

#define STR_SPACE(POS) (rng[POS] >= 0.0 ? " " : "")

int main() {
    double *coef;
    double rng[] = { LOWER_BOUND, UPPER_BOUND };
    double sep = rng[1] - rng[0];
    int i;
    
    sep /= (double) N_SPLITS;
    srand(time(NULL));
    
    coef = init_poly(N_COEF, norm_coef);
    if (coef) {
        rng[1] = eval_poly(rng[0], coef, N_COEF);
        printf("Rnd. Polynom with Normal Coefficients:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly(rng[0], coef, N_COEF);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    /* free_poly(coef); */

    rng[0] = LOWER_BOUND;
    /* coef = init_poly(N_COEF, spike_n_slab_coef); */
    if (coef) {
        rng[1] = eval_poly_n_hard_clip(rng[0], coef, N_COEF, MY_CLIP);
        printf("Rnd. Polynom with Normal Coefficients and HARD Clipping:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly_n_hard_clip(rng[0], coef, N_COEF, MY_CLIP);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    /* free_poly(coef); */

    rng[0] = LOWER_BOUND;
    /* coef = init_poly(N_COEF, spike_n_slab_coef); */
    if (coef) {
        rng[1] = eval_poly_n_soft_clip(rng[0], coef, N_COEF, MY_CLIP);
        printf("Rnd. Polynom with Normal Coefficients and SOFT Clippping:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly_n_soft_clip(rng[0], coef, N_COEF, MY_CLIP);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    free_poly(coef);
    
    rng[0] = LOWER_BOUND;
    coef = init_poly(N_COEF, spike_n_slab_coef);
    if (coef) {
        rng[1] = eval_poly(rng[0], coef, N_COEF);
        printf("Rnd. Polynom with Spike and Slab Coefficients:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly(rng[0], coef, N_COEF);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    /* free_poly(coef); */

    rng[0] = LOWER_BOUND;
    /* coef = init_poly(N_COEF, spike_n_slab_coef); */
    if (coef) {
        rng[1] = eval_poly_n_hard_clip(rng[0], coef, N_COEF, MY_CLIP);
        printf("Rnd. Polynom with Spike and Slab Coefficients and HARD Clipping:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly_n_hard_clip(rng[0], coef, N_COEF, MY_CLIP);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    /* free_poly(coef); */

    rng[0] = LOWER_BOUND;
    /* coef = init_poly(N_COEF, spike_n_slab_coef); */
    if (coef) {
        rng[1] = eval_poly_n_soft_clip(rng[0], coef, N_COEF, MY_CLIP);
        printf("Rnd. Polynom with Spike and Slab Coefficients and SOFT Clippping:\n(%s%.2f -> %s%f)\n", \
               STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        for (i = 0; i < N_SPLITS; i++) {
            rng[0] += sep;
            rng[1] = eval_poly_n_soft_clip(rng[0], coef, N_COEF, MY_CLIP);
            printf("(%s%.2f -> %s%f)\n", \
                   STR_SPACE(0), rng[0], STR_SPACE(1), rng[1]);
        }
        printf("\n");
    }
    free_poly(coef);
    return 0;
}
#endif

