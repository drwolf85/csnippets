#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/**
 * @brief Conditional Poisson Sampling via Rejection Method
 *
 * @param pik Pointer to a vector of inclusion probabilities
 * @param N Length of the vector of inclusion probabilities
 * @param n Sample size
 *
 * @return Pointer to a vector of boolean values of size N
 */
bool * cpois_reject(double *pik, size_t N, size_t n) {
	bool *smp = NULL;
	double u;
	size_t i = 0, c = 0;
	size_t pc = 0;
        for (i = 0; i < N; i++) pc += (size_t) (pik[i] > 0.0);
        if (n > N || n > pc) return smp;
	smp = (bool *) calloc(N, sizeof(bool));
	if (__builtin_expect(smp != NULL, 1)) {
		while (c < n) {
			i %= N;
			u = 0.5 + (double) arc4random();
			u /= (double) (1ULL << 32ULL);
			if (u <= pik[i] && smp[i] == false) {
				smp[i] = true;
				c++;
			}
			i++;
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
	double sm = 0.0;
	for (i = 0; i < N; i++) sm += pik[i];
	sm = (double) n / sm;
	for (i = 0; i < N; i++) pik[i] *= sm;
	printf("Inclusion probabilities:\n");
	for (i = 0; i < N; i++) printf("%f ", pik[i]);
	bool *res = cpois_reject(pik, N, n);
	if (__builtin_expect(res != NULL, 1)) {
		printf("\nSampled if equal to one:\n");
		for (i = 0; i < N; i++) printf("%u ", (unsigned) res[i]);
		free(res);
	}
	printf("\n");
	return 0;
}
#endif

