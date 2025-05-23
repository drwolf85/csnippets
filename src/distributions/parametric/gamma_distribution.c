#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RUNIF01 ((double) arc4random() / (double) ((1ULL << 32ULL) - 1ULL))
#define REXP1 (-log(RUNIF01))

extern double dgamma(double x, double alpha, double beta) {
	double res = nan("");
	if (alpha > 0.0 && beta > 0.0 && x > 0.0) {
		res = (alpha - 1.0) * log(x) - x / beta;
		res -= alpha * log(beta) + lgamma(alpha);
		res = exp(res);
	}
	return res;
}

static inline double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = arc4random();
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

extern double rgamma(double alpha, double beta) {
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

#ifdef DEBUG
#define N 80
int main() {
	unsigned i;
	double g;
	printf("Gamma density f(1, 1, 1) = %f\n", dgamma(1.0,1.0,1.0));
	for (i = 0; i < N; i++) {
		g = rgamma(10.999, 0.5);
		printf("%g%s", g, (i + 1) % 8 ? " " : "\n");
	}
	return 0;
}

#endif
