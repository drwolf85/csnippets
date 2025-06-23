#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * Generation of Binomial(n, p) random numbers
 *
 * @param n Number of trials
 * @param prob Probability of success
 *
 * @return size_t
 * */
static inline size_t rbinom(size_t n, double prob) {
    size_t i;
    size_t z = 0;
    for (i = 0; i < n; i++) {
        z += (size_t) (ldexp((double) arc4random(), -32) < prob);
    }
    return z;
}


/**
 * @brief Multinomial Design Sampling
 *
 * @param pik Pointer to a vector of inclusion probabilities
 * @param N Length of the vector of inclusion probabilities
 * @param n Sample size
 *
 * @return Pointer to a vector with N integer values
 */
size_t * Multinomial_Design(double *pik, size_t N, size_t n) {
	size_t *smp = NULL;
	size_t sm = 0;
	double smu = 0.0;
	size_t i = 0;
	size_t pc = 0;
	for (i = 0; i < N; i++) pc += (size_t) (pik[i] > 0.0);
	if (n > N || n > pc) return smp;
	smp = (size_t *) calloc(N, sizeof(size_t));
	if (__builtin_expect(smp != NULL, 1)) {
#ifdef DEBUG
		printf("Inclusion probabilities:\n");
		for (i = 0; i < N; i++) printf("%f ", pik[i]);
		printf("\n");
#endif
		for (i = 0; i < N && sm < n; i++) {
			smp[i] = rbinom(n - sm, pik[i] / ((double) n - smu));
			smu += pik[i];
			sm += smp[i];
		}
	}
	return smp;
}

#ifdef DEBUG
int main(void) {
	double pik[] = {0.9, 0.1, 0.2, 0.7, 0.9, 0.8, 0.3};
	size_t const N = 7;
	size_t const n = 3;
	size_t i; 
	size_t *res = NULL;
	double nrm = 0.0;
	for (i = 0; i < N; i++) nrm += pik[i];
	nrm = (double) n / nrm;
	for (i = 0; i < N; i++) pik[i] *= nrm;
	res = Multinomial_Design(pik, N, n);
	if (__builtin_expect(res != NULL, 1)) {
		printf("Multinomial Design Sample:\n");
		for (i = 0; i < N; i++) printf("%lu ", res[i]);
	}
	if (__builtin_expect(res != NULL, 1)) free(res);
	printf("\n");
	return 0;
}
#endif

