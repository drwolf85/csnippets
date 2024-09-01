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

uint32_t * univar_counts(uint32_t *res, uint32_t *x_var, uint32_t n) {
	uint32_t i, j, sz = 1;
	uint32_t *count;
	uint32_t *tmpv = (uint32_t *) malloc(VEC_SIZE);
	if (tmpv) {
		memcpy(tmpv, x_var, VEC_SIZE);
		qsort(tmpv, n, sizeof(uint32_t), u32_cmp);
		#pragma omp parallel for simd private(i) reduction(+ : sz)
		for (i = 1; i < n; i++) {
			sz += (uint32_t) (tmpv[i] != tmpv[i - 1]);
		}
		*res = sz;
		count = (uint32_t *) calloc(sz, sizeof(uint32_t));
		if (count) {
			j = 0;
			count[j] = 1;
			for (i = 1; i < n; i++) {
				sz = (tmpv[i] == tmpv[i - 1]);
				count[j] += sz;
				j += !sz;
				count[j] += !sz;
			}
		}
	}
	free(tmpv);
	return count;
}

#ifdef DEBUG
int main() {
  uint32_t x[] = {8, 4, 5, 6, 1, 5, 0, 7, 3, 1, 2, 3, 5};
  uint32_t const n = 13;
  uint32_t i, m;
  uint32_t *ctv = univar_counts(&m, x, n);
  if (ctv) {
    printf("There are %u categorical values\n", m);
    for (i = 0; i < m; i++) {
      printf("Count of category \"%d\": %u\n", i, ctv[i]);
    }
  }
  free(ctv);
  return 0;
}
#endif

