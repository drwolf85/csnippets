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
    double *x;
    double y;
} point;

unsigned locpoly(point *pred, unsigned npred, point *obs, unsigned nobs, unsigned szx, double (*kern)(double), double smooth) {
    unsigned i, j, k;
    double x, a, b, w, xm , ym, x2, xy, nrm, val;
    if (npred > 0 && nobs > 0 && szx > 0 && pred && obs) {
        //#pragma omp parallel for simd private(j, x, a, b, w, xm , ym, x2, xy, nrm, val)
        for (i = 0; i < npred; i++) {
            xm = ym = x2 = xy = nrm = 0.0;
            for (j = 0; j < nobs; j++) {
                /* BEGIN: Euclidean normalization... but other ways depend on the metric space at hand */
                val = 0.0;
                for (k = 0; k < szx; k++) {
                    x = obs[j].x[k] - pred[i].x[k];
                    val += x * x;
                }
                x = sqrt(val / (double) szx);
                /* END: Euclidean normalization */
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
double myfun(double *x) {
    double res = 1.0 + exp(-0.025 + 0.5 * x[0] + 0.2 * sin(x[1] + 1.23));
    res = 1.0 / res;
    return (1.0 - res) * res;
}

int main() {
    unsigned i, k; 
    unsigned const szx = 2;
    unsigned const nobs = 2001;
    unsigned const npred = 5;
    point *my_obs = (point *) calloc(nobs, sizeof(point));
    point *my_pred = (point *) calloc(npred, sizeof(point));
    if (my_obs && my_pred) {
        /* Initialize observed data */
        for (i = 0; i < nobs; i++) {
            my_obs[i].x = (double *) calloc(szx, sizeof(double));
            if (my_obs[i].x) {
                my_obs[i].x[0] = -5.0 + (double) i * (10.0 / (double) nobs);
                my_obs[i].x[1] = 2.0 - (double) i * (4.0 / (double) nobs);
            }
            my_obs[i].y = myfun(my_obs[i].x);
        }
        /* Initialize points for predictions */
        for (i = 0; i < npred; i++) {
            my_pred[i].x = (double *) calloc(szx, sizeof(double));
            if (my_pred[i].x) {
                my_pred[i].x[0] = -5.01 + 0.99 * (double) i * (10.0 / (double) npred);
                my_pred[i].x[1] = 2.01 - 0.99 * (double) i * (4.0 / (double) npred);
            }
            my_pred[i].y = nan("");
        }
        i = locpoly(my_pred, npred, my_obs, nobs, szx, laplacian_kernel, 2.0);
        if (i == npred) {
            printf("Linear Laplacian smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("Smooth: %f (true: %f)\n", my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
        i = locpoly(my_pred, npred, my_obs, nobs, szx, silverman_kernel, 2.0);
        if (i == npred) {
            printf("Linear Silverman smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("Smooth: %f (true: %f)\n", my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
        i = locpoly(my_pred, npred, my_obs, nobs, szx, gaussian_kernel, 2.0);
        if (i == npred) {
            printf("Linear Gaussian smoother:\n");
            for (i = 0; i < npred; i++) {
                printf("Smooth: %f (true: %f)\n", my_pred[i].y, myfun(my_pred[i].x)); 
            }
        }
        for (i = 0; i < nobs; i++) free(my_obs[i].x);
        for (i = 0; i < npred; i++) free(my_pred[i].x);
    }
    free(my_obs);
    free(my_pred);
    return 0;
}
#endif