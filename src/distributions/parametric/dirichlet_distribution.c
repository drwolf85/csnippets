#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RUNIF01 ((double) rand() / (double) RAND_MAX)
#define REXP1 (-log(RUNIF01))

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

static inline double halpha(double z, double alpha, double lc) {
	return exp(lc - z - exp(-z / alpha));
}

static inline double etalpha(double z, double alpha, double wa, double la, double lc) {
	double res;
	if (z >= 0.0) {
		res = exp(lc - z);
	}
	else {
		res = wa * la * exp(lc + la * z);
	}
	return res;
}

static inline double rgamma(double alpha, double beta) {
	double res = 0.0, u, z;
	if (alpha >= 10000.0) {
        	res = rnorm(alpha, sqrt(alpha));
	}
	else {
		while (alpha >= 1.0) {
			res += REXP1;
			alpha -= 1.0;
		}
		if (alpha > 0.0) {
			double const lc = - lgamma(alpha + 1.0);
			double const wa = exp(-1.0) * alpha / (1.0 - alpha);
			double const ra = 1.0 / (1.0 + wa);
			double const la = 1.0 / alpha - 1.0;
			for (;;) {
				u = RUNIF01;
				if (u <= ra) {
					z = -log(u / ra);
				}
				else {
					z = log(RUNIF01) / la;
				}
				if ((halpha(z, alpha, lc) / etalpha(z, alpha, wa, la, lc)) > RUNIF01) break;
			}
			res += exp(- z / alpha);
		}
	}
	return res * beta;
}

extern double * rdirichlet(double *alpha, unsigned p) {
	double *a = (double *) calloc(p, sizeof(double));
	double den = 0.0;
	unsigned i;
	for (i = 0; i < p; i++) den += a[i] = rgamma(alpha[i], 1.0);
	den = 1.0 / den;
	for (i = 0; i < p; i++) a[i] *= den;
	return a;
}

#ifdef DEBUG
#define N 10
int main() {
	unsigned i, j, p = 8;
	double par[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.5, 3.4};
	double *g;
	srand(time(NULL));
	for (i = 0; i < N; i++) {
		g = rdirichlet(par, p);
		if (g) for (j = 0; j < p; j++) printf("%.4f%s", g[j], (j + 1) % 8 ? " " : "\n");
		free(g);
	}
	return 0;
}

#endif
