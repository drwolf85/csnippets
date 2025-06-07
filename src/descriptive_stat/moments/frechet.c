#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MY_EPS 1e-8
#define MAXIT 1000
/**
 * @brief Frechet variance
 *
 * @param mean Pointer to a data structure representing the frechet mean
 * @param y Pointer to an array of data structures used as input data
 * @param n Number of data points (i.e., length of `y`)
 * @param sz_data Size of the data structure to process
 * @param dst Pointer to a distance/metric function
 *
 * @return double
 */
extern inline double frechet_var(void *mean, void *y, size_t n, size_t const sz_data,
		                 double (*dst)(void const *, void const *)) {
	size_t i;
	double tmp, ans = nan("");
	if (mean && y && n && sz_data && dst) {
		ans = 0.0;
		for (i = 0; i < n * sz_data; i += sz_data) {
			tmp = dst((void *) &((int8_t *) y)[i], mean);
			ans += tmp * tmp;
		}
		ans /= (double) n;
	}
	return ans;
}

/**
 * @brief Frechet mean
 *
 * @param mean Pointer to a data structure where to save the mean
 * @param y Pointer to an array of data structures used as input data
 * @param n Number of data points (i.e., length of `y`)
 * @param sz_data Size of the data structure to process
 * @param dst Pointer to a distance/metric function
 * @param oracle Pointer to an oracle function used to minimize `dst()`
 */
void frechet_mean(void *mean, void *y, size_t n, size_t sz_data,
		  double (*dst)(void const *, void const *), 
		  void * (*oracle)(void *, void *, size_t, size_t)) {
	void *tom, *olly;
	double dol, dtm;
	size_t cnt = 0;
	olly = oracle(mean, y, n, sz_data);
	tom = oracle(mean, y, n, sz_data);
	do {
		dol = frechet_var(olly, y, n, sz_data, dst);
		dtm = frechet_var(tom, y, n, sz_data, dst);
		if (dtm < dol) {
			memcpy(mean, tom, sz_data);
			if (__builtin_expect(olly != NULL, 1)) {
				free(olly);
				olly = oracle(mean, y, n, sz_data);
			}
		}
		else {
			memcpy(mean, olly, sz_data);
			if (__builtin_expect(tom != NULL, 1)) {
				free(tom);
				tom = oracle(mean, y, n, sz_data);
			}
		}
		cnt++;
	} while (fabs(dol - dtm) > MY_EPS && cnt < MAXIT);
	if (__builtin_expect(olly != NULL, 1)) free(olly);
	if (__builtin_expect(tom != NULL, 1)) free(tom);
}

#ifdef DEBUG
#define N 5

double data_y[N] = {0.1, 0.2, 0.3, 0.4, 0.5};

double simpleL1(void const *aa, void const *bb) {
	double *a = (double *) aa;
	double *b = (double *) bb;
	double res = nan("");
	if (__builtin_expect(a && b, 1)) {
		res = fabs(*a - *b);
	}
	return res;
}

#define MY_STEP 0.1
void * my_oracle(void *mean, void *data, size_t n, size_t sz) {
	double *guess = (double *) calloc(1, sz);
	double *m = (double *) mean;
	double *y = (double *) data;
	size_t i;
	double const inv = MY_STEP / (double) n;
	if (mean && data && n && sz && guess) {
		*guess = *m;
		for (i = 0; i < n; i++) {
			*guess += inv * (0.5 + 0.5 * ldexp((double) arc4random(), -32)) \
				      * (double) ((y[i] > *m) - (y[i] < *m));
		}
	}
	/* printf("Guess: %f\n", *guess); */
	return (void *) guess;
}

int main(void) {
	double m = data_y[arc4random() % N], v = 0.0;
	frechet_mean((void *) &m, (void *) data_y, \
		     N, sizeof(double), simpleL1, my_oracle);
	v = frechet_var((void *) &m, (void *) data_y, N, sizeof(double), simpleL1);
	printf("Mean: %f\n", m);
	printf("Variance: %f\n", v);
	return 0;
}

#endif

