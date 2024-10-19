#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_IT 3000

/* Prototype of Empirical Likelihood method for outlier detection */

static double mn;
#define QT_NET_VAL 0.3882244831294644482611

static inline void get_logs(double *err, unsigned const n, bool verbose) {
	unsigned i;
	double ee = fabs(err[0]);
	mn = ee;
	for (i = 1; i < n; i++) {
 		ee = fabs(err[i]);
 		mn += (double) (ee > mn) * (ee - mn);
	}
	if (verbose) printf("Worse absolute error: %.4f\n", mn);
	mn = -log1p(mn * QT_NET_VAL);
	return;
}

/* The updating formulas in the two functions that follow
   are correctly derived from eq. (15) in DOI:10.3390/stats7040073 */

static inline double w(double l, double p, double par) {
	double res = l + p * mn;
	return 1.0 / res;
}

static inline void update(double *l, double *p, double par) {
	double wt;
	int j = 0;
	do {
		wt = w(*l, *p, par);
		*l += wt - 1.0;
		*p += wt * mn - log(par);
	} while (j++ < MAX_IT);
}

extern inline double EL(double *l, double *p, double par) {
	double wv = 0.0;
	update(l, p, par);
	wv = w(*l, *p, par);
	wv *= (double) (wv > 0.0);
	return fabs(wv);
}

#ifdef DEBUG

#define K 3
#define M 11

/* Array of standardized residuals */
static double eps_vec[K] = {2.5759, 0.85, 0.84};

int main() {
	double par, wt;
	double l, p;
	double const im = 1.0 / (double) M;
	int i, j = 0;
	double c, ll = 0.0, ul = 0.0;
	get_logs(eps_vec, K, true);
	for (i = 0; i <= M; i++) {
		l = 1.0;
		p = 0.0;
		par = 1e-3 + 0.998 * im * (double) i;
		wt = EL(&l, &p, par);
		ll += wt * (double) (par < 0.5);
		ul += wt * (double) (par >= 0.5);
		printf("EL(%.4f) = %.4f\n", par, wt);
	}
	c = 1.0 / (ll + ul);
	printf("Pr(off) = %.4f, Pr(ok) = %.4f\n", ll * c, ul * c);
	return 0;
}
#endif

