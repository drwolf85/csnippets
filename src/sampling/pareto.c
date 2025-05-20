#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

typedef struct pareto_rank {
	double q;
	size_t i;
} pareto_rank;

static int cmp_pranks(void const *aa, void const *bb) {
	pareto_rank a = *(pareto_rank *) aa;
	pareto_rank b = *(pareto_rank *) bb;
	return (int) (a.q > b.q) * 2  - 1;
}

/**
 * @brief Pareto Sampling
 *
 * @param pik Pointer to a vector of inclusion probabilities
 * @param N Length of the vector of inclusion probabilities
 * @param n Sample size
 *
 * @return Pointer to a vector of boolean values of size N
 */
bool * pareto(double *pik, size_t N, size_t n) {
	double u;
#ifdef DEBUG
	double sum = 0.0;
#endif
	size_t i = 0, pc = 0;
	bool *smp = NULL;
	pareto_rank *q = NULL;
	for (i = 0; i < N; i++) pc += (size_t) (pik[i] > 0.0);
	if (n > N || n > pc) return smp;
	smp = (bool *) calloc(N, sizeof(bool));
	q = (pareto_rank *) malloc(N * sizeof(pareto_rank));
	if (__builtin_expect(smp && q, 1)) {
#ifdef DEBUG
		for (i = 0; i < N; i++) sum += pik[i];
		sum = (double) n / sum;
#endif
		for (i = 0; i < N; i++) {
#ifdef DEBUG
			pik[i] *= sum;
#endif
			if (pik[i] <= 0.0) { 
				q[i].q = INFINITY;
			}
			else if (pik[i] >= 1.0) {
				q[i].q = 0.0;
			}
			else {
				u = 0.5 + (double) arc4random();
				u /= (double) (1ULL << 32ULL);
				q[i].q = u * (1.0 - pik[i]);
				q[i].q /= pik[i] * (1.0 - u);
			}
			q[i].i = i;
		}
#ifdef DEBUG
		printf("Inclusion probabilities:\n");
		for (i = 0; i < N; i++) printf("%f ", pik[i]);
		printf("\nPareto ranks:\n");
		for (i = 0; i < N; i++) printf("%f ", q[i].q);
		printf("\n");
#endif
		qsort(q, N, sizeof(pareto_rank), cmp_pranks);
		for (i = 0; i < n; i++) {
			smp[q[i].i] = true;
		}
	}
	if (__builtin_expect(q != NULL, 1)) free(q);
	return smp;
}

#ifdef DEBUG
int main(void) {
	double pik[] = {0.9, 0.1, 0.2, 0.7, 0.9, 0.8, 0.3};
	size_t const N = 7;
	size_t const n = 3;
	size_t i; 
	bool *res;
	res = pareto(pik, N, n);
	if (__builtin_expect(res != NULL, 1)) {
		printf("Sampled if equal to one:\n");
		for (i = 0; i < N; i++) printf("%u ", (unsigned) res[i]);
	}
	if (__builtin_expect(res != NULL, 1)) free(res);
	printf("\n");
	return 0;
}
#endif

