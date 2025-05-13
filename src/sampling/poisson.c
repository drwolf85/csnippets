#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/**
 * @brief Poisson Sampling
 *
 * @param pik Pointer to a vector of inclusion probabilities
 * @param N Length of the vector of inclusion probabilities
 * @param n Sample size
 *
 * @return Pointer to a vector of boolean values of size N
 */
bool * poisson(double *pik, size_t N, size_t n) {
	bool *smp = NULL;
	double sm = 0.0;
	size_t i = 0;
	size_t pc = 0;
	for (i = 0; i < N; i++) pc += (size_t) (pik[i] > 0.0);
	if (n > N || n > pc) return smp;
	smp = (bool *) calloc(N, sizeof(bool));
	if (__builtin_expect(smp != NULL, 1)) {
		for (i = 0; i < N; i++) sm += pik[i];
		sm = (double) n / sm;
		for (i = 0; i < N; i++) pik[i] *= sm;
#ifdef DEBUG
		printf("Inclusion probabilities:\n");
		for (i = 0; i < N; i++) printf("%f ", pik[i]);
		printf("\n");
#endif
		for (i = 0; i < N; i++) {
			sm = 0.5 + (double) rand(); 
			sm /= 1.0 + (double) RAND_MAX;
			if (sm <= pik[i]) {
				smp[i] = true;
			}
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
	srand(time(NULL));
	bool *res = poisson(pik, N, n);
	if (__builtin_expect(res != NULL, 1)) {
		printf("Sampled if equal to one:\n");
		for (i = 0; i < N; i++) printf("%u ", (unsigned) res[i]);
	}
	if (__builtin_expect(res != NULL, 1)) free(res);
	printf("\n");
	return 0;
}
#endif

