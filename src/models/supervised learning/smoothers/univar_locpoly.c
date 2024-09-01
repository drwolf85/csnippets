#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <omp.h>

static double laplacian_kernel(double x) {
    return 0.5 * exp(-fabs(x));
}

static double silverman_kernel(double x) {
    double tmp = fabs(x) * M_SQRT1_2;
    return 0.5 * exp(-tmp) * sin(tmp + M_PI_4);
}

static double gaussian_kernel(double x) {
    x = exp(-0.5 * x * x);
    return x / sqrt(2.0 * M_PI);
}

typedef struct point_str {
    double x;
    double y;
} point;

unsigned univar_locpoly(point *pred, unsigned npred, point *obs, unsigned nobs, double (*kern)(double), double smooth) {
    unsigned i, j;
    double x, a, b, w, xm , ym, x2, xy, nrm;
    if (npred > 0 && nobs > 0 && pred && obs) {
        //#pragma omp parallel for simd private(j, x, a, b, w, xm , ym, x2, xy, nrm)
        for (i = 0; i < npred; i++) {
            xm = ym = x2 = xy = nrm = 0.0;
            for (j = 0; j < nobs; j++) {
                x = obs[j].x - pred[i].x;
                nrm += w = (kern(x * smooth) * smooth);
                a = x * w;
                xm += a;
                x2 += x * a;
                b = obs[j].y * w;
                ym += b;
                xy += b * x;
            }
            nrm = 1.0 / nrm;
            xm *= nrm;
            ym *= nrm;
            x2 *= nrm;
            xy *= nrm;
            b = xy - xm * ym;
            b /= x2 - xm * xm;
            pred[i].y = ym - b * xm;
        }
    }
    else {
        npred = 0;
        free(pred);
    }
    return npred;
}

#ifdef DEBUG
double myfun(double x) {
    return tanh(-0.025 + 0.05 * x + 0.2 * sin(x + 1.23));
}

int main() {
    unsigned i; 
    unsigned const nobs = 2001;
    unsigned const npred = 5;
    point *my_obs = (point *) calloc(nobs, sizeof(point));
    point *my_pred = (point *) calloc(npred, sizeof(point));
    if (my_obs && my_pred) {
        /* Initialize observed data */
        for (i = 0; i < nobs; i++) {
            my_obs[i].x = -5.0 + (double) i * (10.0 / (double) nobs);
            my_obs[i].y = myfun(my_obs[i].x);
        }
        /* Initialize points for predictions */
        for (i = 0; i < npred; i++) {
            my_pred[i].x = -5.01 + 0.99 * (double) i * (10.0 / (double) npred);
            my_pred[i].y = nan("");
        }
        i = univar_locpoly(my_pred, npred, my_obs, nobs, laplacian_kernel, 2.0);
        if (i == npred) {
            printf("Linear Laplacian smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("%f -> %f (true: %f)\n", my_pred[i].x, my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
        i = univar_locpoly(my_pred, npred, my_obs, nobs, silverman_kernel, 2.0);
        if (i == npred) {
            printf("Linear Silverman smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("%f -> %f (true: %f)\n", my_pred[i].x, my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
        i = univar_locpoly(my_pred, npred, my_obs, nobs, gaussian_kernel, 2.0);
        if (i == npred) {
            printf("Linear Gaussian smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("%f -> %f (true: %f)\n", my_pred[i].x, my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
    }
    free(my_obs);
    free(my_pred);
    return 0;
}
#endif