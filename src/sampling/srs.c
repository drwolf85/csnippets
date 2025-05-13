#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/**
 * @brief Simple Random Sampling With Replacement (SRSWR)
 *
 * @param N Number of population units (i.e., population size)
 * @param n Number of samples to select at random
 *
 * @return Pointer to a vector of `n` sample IDs  
 */
size_t * srswr(size_t N, size_t n) {
	size_t *smp = NULL;
	size_t i = 0;
	double u;
	smp = (size_t *) calloc(n, sizeof(size_t));
        if (__builtin_expect(smp != NULL, 1)) {
		for (; i < n; i++) {
			u = (double) rand() / (0.5 + (double) RAND_MAX);
			smp[i] = (size_t) floor(u * (double) N);
		}
	}
	return smp;
}

/**
 * @brief Selection probability for Simple Random Sampling Without Replacement
 *
 * @param r Number corresponding to the r-th iteration
 * @param N Number of population units (i.e., population size)
 *
 * @return double
 */
static inline double prob_srswor(size_t r, size_t N) {
	size_t cnt = N - r + 1;
	return 1.0 / (double) cnt;
}

/**
 * @brief Simple Random Sampling Without Replacement (SRSWOR)
 *
 * @param N Number of population units (i.e., population size)
 * @param n Number of samples to randomly select 
 *
 * @return Pointer to a vector of boolean values of size N
 */
bool * srswor(size_t N, size_t n) {
	bool *smp = NULL;
	size_t i = 0, c = 0;
	double pr, u;
        if (n > N) return smp;
	smp = (bool *) calloc(N, sizeof(bool));
        if (__builtin_expect(smp != NULL, 1)) {
		while (c < n) {
			i %= N;
			pr = prob_srswor(c + 1, N);
			u = 0.5 + (double) rand();
			u /= 1.0 + (double) RAND_MAX;
			if (u <= pr && smp[i] == false) {
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
        size_t const N = 7;
        size_t const n = 3;
        size_t i;
        srand(time(NULL));
        size_t *vec = srswr(n, N);
	bool *res = srswor(N, n);
        if (__builtin_expect(res && vec, 1)) {
                printf("SRSWR (samples IDs are shown below):\n");
                for (i = 0; i < N; i++) printf("%lu ", vec[i]);
                printf("\nSRSWOR (samples are equal to one):\n");
                for (i = 0; i < N; i++) printf("%u ", (unsigned) res[i]);
        }
        if (__builtin_expect(res != NULL, 1)) free(res);
        if (__builtin_expect(vec != NULL, 1)) free(vec);
        printf("\n");
        return 0;
}
#endif
