#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/**
 * @brief Systematic Sampling
 *
 * @param pik Pointer to a vector of inclusion probabilities
 * @param N Length of the vector of inclusion probabilities
 * @param n Sample size
 *
 * @return Pointer to a vector of boolean values of size N
 */
bool * systematic(double *pik, size_t N, size_t n) {
	bool *smp = NULL;
	double a, b, sm = 0.0;
	size_t i = 0;
	size_t pc = 0;
	for (i = 0; i < N; i++) pc += (size_t) (pik[i] > 0.0);
	if (__builtin_expect(n > N || n > pc, 0)) return smp;
	smp = (bool *) calloc(N, sizeof(bool));
	if (__builtin_expect(smp != NULL, 1)) {
#ifdef DEBUG
		for (i = 0; i < N; i++) sm += pik[i];
		sm = (double) n / sm;
		for (i = 0; i < N; i++) pik[i] *= sm;
		printf("Inclusion probabilities:\n");
		for (i = 0; i < N; i++) printf("%f ", pik[i]);
		printf("\n");
#endif
/*		sm = 0.5 + (double) arc4random(); 
		sm /= (double) (1ULL << 32ULL); */
		sm = ldexp((double) arc4random(), -32);
		a = -sm;
		for (i = 0; i < N; i++) {
			b = a;
			a += pik[i];
			if ((int64_t) floor(a) != (int64_t) floor(b)) {
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
	bool *res = systematic(pik, N, n);
	if (__builtin_expect(res != NULL, 1)) {
		printf("Sampled if equal to one:\n");
		for (i = 0; i < N; i++) printf("%u ", (unsigned) res[i]);
	}
	if (__builtin_expect(res != NULL, 1)) free(res);
	printf("\n");
	return 0;
}
#endif

