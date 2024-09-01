#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define VEC_SIZE (n * sizeof(uint32_t))

int u32_cmp(const void *aa, const void *bb) {
  uint32_t a = *(uint32_t *) aa;
  uint32_t b = *(uint32_t *) bb;
  return (int) (a > b) * 2 - 1;
}

uint32_t n_kat(uint32_t *x_var, int n) {
	uint32_t count = 1, i = 0;
	uint32_t *tmpv = (uint32_t *) malloc(VEC_SIZE);
	if (tmpv) {
		memcpy(tmpv, x_var, VEC_SIZE);
		qsort(tmpv, n, sizeof(uint32_t), u32_cmp);
		#pragma omp parallel for simd private(i) reduction(+ : count)
		for (i = 1; i < n; i++) {
		  count += (uint32_t) (tmpv[i] != tmpv[i - 1]);
		}
	}
	free(tmpv);
	return count;
}

#ifdef DEBUG
int main() {
  uint32_t x[] = {8, 4, 5, 4, 6, 1, 5, 0, 7, 3, 1, 2, 3};
  int const n = 13;
  uint32_t test = n_kat(x, n);
  printf("There are %u categorical values\n", test);
  return 0;
}
#endif

