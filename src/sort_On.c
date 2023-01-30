#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/**
@brief "Negative copy" 
@param dest Destination vector (double)
@param src Source vector (double)
@param n length of the two vectors above (size_t)
@param sgn Character determining how to treat negative numbers
*/
void negcpy(double *dest, double *src, size_t n, char sgn) {
    size_t i;
    if (sgn) { /* Part to execute at the end when `sgn != 0` */
        for (i = 0; i < n; i++) {
            dest[i] = -src[i];
            if (src[i] > 0.0)
                dest[i] = 1.0 / dest[i];
        }
    }
    else { /* Part to execute at the beginning when `sgn == 0` */
        for (i = 0; i < n; i++) {
            dest[i] = -src[i];
            if (src[i] < 0.0)
                dest[i] = 1.0 / dest[i];
        }
    }
}

/**
@brief Sorting a vector of real number (from smallest to largest)
@param x Vector of real numbers (double)
@param n length of the vector above (size_t)
*/
void sort_reals(double *x, size_t n) {
	double *s, *y;
	uint8_t c;
	uint64_t v;
	size_t i, h[256], ch[256];
	s = (double *) malloc(n * sizeof(double));
	y = (double *) malloc(n * sizeof(double));

	if (s && y) {
		//memcpy(s, x, n * sizeof(double));
        negcpy(s, x, n, 0);
		for (c = 0; c < sizeof(double); c++) {
			memset(h,  0, 256 * sizeof(size_t));
			memset(ch, 0, 256 * sizeof(size_t));
			/* build histogram counts */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				h[v]++;
			}
			/* adjust starting positions */
			for (i = 1; i < 256; i++) {
				ch[i] = ch[i-1] + h[i-1];
			}
			/* sort values */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				y[ch[v]] = s[i];
				ch[v]++;
            }
			/* Copy to sorted vector */
			memcpy(s, y, n * sizeof(double));
		}
		//memcpy(x, s, n * sizeof(double));
        negcpy(x, s, n, 1);
	}
	free(s);
	free(y);
}

/**
@brief Sorting a vector of real number (from 
       smallest to largest) with indexes
@param x Vector of real numbers (double)
@param idx Vector of indices (integer numbers, size_t)
@param n length of the vector above (size_t)
*/
void sort_reals_wid(double *x, size_t *idx, size_t n) {
	double *s, *y;
	uint8_t c;
	uint64_t v;
	size_t i, h[256], ch[256], *sid;
	sid = (size_t *) malloc(n * sizeof(size_t));
	s = (double *) malloc(n * sizeof(double));
	y = (double *) malloc(n * sizeof(double));

	if (s && y && sid) {
		//memcpy(s, x, n * sizeof(double));
    	negcpy(s, x, n, 0);
		for (c = 0; c < sizeof(double); c++) {
			memset(h,  0, 256 * sizeof(size_t));
			memset(ch, 0, 256 * sizeof(size_t));
			/* build histogram counts */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				h[v]++;
			}
			/* adjust starting positions */
			for (i = 1; i < 256; i++) {
				ch[i] = ch[i-1] + h[i-1];
			}
			/* sort values */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				y[ch[v]] = s[i];
        		sid[ch[v]] = idx[i];
				ch[v]++;
      		}
			/* Copy to sorted vectors */
			memcpy(s, y, n * sizeof(double));
      		memcpy(idx, sid, n * sizeof(size_t));
		}
		//memcpy(x, s, n * sizeof(double));
    	negcpy(x, s, n, 1);
	}
	free(sid);
	free(s);
	free(y);
}

/**
@brief Sorting a vector of non-negative numbers (from smallest to largest)
@param x Vector of real non-negative numbers (double)
@param n length of the vector above (size_t)
*/
void sort_pos(double *s, size_t n) {
	double *y;
	uint8_t c;
	uint64_t v;
	y = (double *) malloc(n * sizeof(double));
	size_t i, h[256], ch[256];

	if (y) {
		for (c = 0; c < sizeof(double); c++) {
			memset(h,  0, 256 * sizeof(size_t));
			memset(ch, 0, 256 * sizeof(size_t));
			/* build histogram counts */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				h[v]++;
			}
			/* adjust starting positions */
			for (i = 1; i < 256; i++) {
				ch[i] = ch[i-1] + h[i-1];
			}
			/* sort values */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				y[ch[v]] = s[i];
				ch[v]++;
      		}
			/* Copy to sorted vector */
			memcpy(s, y, n * sizeof(double));
		}
	}
	free(y);
}

/**
@brief Sorting a vector of non-negative number (from 
       smallest to largest) with indexes
@param x Vector of non-negative numbers (double)
@param idx Vector of indices (integer numbers, size_t)
@param n length of the vector above (size_t)
*/
void sort_pos_wid(double *s, size_t *idx, size_t n) {
	double *y;
	uint8_t c;
	uint64_t v;
	size_t i, h[256], ch[256], *sid;

	sid = (size_t *) malloc(n * sizeof(size_t));
	y = (double *) malloc(n * sizeof(double));

	if (y && sid) {
		for (c = 0; c < sizeof(double); c++) {
			memset(h,  0, 256 * sizeof(size_t));
			memset(ch, 0, 256 * sizeof(size_t));
			/* build histogram counts */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				h[v]++;
			}
			/* adjust starting positions */
			for (i = 1; i < 256; i++) {
				ch[i] = ch[i-1] + h[i-1];
			}
			/* sort values */
			for (i = 0; i < n; i++) {
				v = *(uint64_t *) &s[i];
				v >>= 8 * c;
				v &= 255;
				y[ch[v]] = s[i];
        		sid[ch[v]] = idx[i];
				ch[v]++;
			}
			/* Copy to sorted vector */
			memcpy(s, y, n * sizeof(double));
			memcpy(idx, sid, n * sizeof(size_t));
		}
	}
	free(sid);
	free(y);
}

/* Test program for the functions above */
# define N 5
int main() {
	double x[N] = {84.7, -1.2, -45.2, 4.5, 0.2};
	double s[N] = {84.7, 1.2, 45.2, 4.5, 0.2};
	double y[N] = {84.7, -1.2, -45.2, 4.5, 0.2};
	double z[N] = {84.7, 1.2, 45.2, 4.5, 0.2};
	size_t idx[N];

	/* Sort real numbers */
	sort_reals(y, N);
	for (int i = 0; i < N; i++) {
		idx[i] = i;
		printf("%f ", y[i]);
	}
	printf("\n");
	/* Sort non-negative numbers */
	sort_pos(z, N);
	for (int i = 0; i < N; i++) {
		printf("%f ", z[i]);
	}
	printf("\n");
	/* Sort real numbers with indices */
	sort_reals_wid(x, idx, N);
	for (int i = 0; i < N; i++) {
		printf("%f (%lu) ", x[i], idx[i]);
    idx[i] = i;
	}
	printf("\n");
	/* Sort non-negative numbers with indices */
	sort_pos_wid(s, idx, N);
	for (int i = 0; i < N; i++) {
		printf("%f (%lu) ", s[i], idx[i]);
	}
	printf("\n");

	return 0;
}
