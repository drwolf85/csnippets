#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../data structures/uni_table.h"

#define VEC_SIZE (n * sizeof(uint32_t))

int u32_cmp(const void *aa, const void *bb) {
  uint32_t a = *(uint32_t *) aa;
  uint32_t b = *(uint32_t *) bb;
  return (int) (a > b) * 2 - 1;
}

uni_table unitabl (uint32_t *x_var, uint32_t n) {
	uint32_t i, j, sz = 1;
	uni_table tbl;
	uint32_t *tmpv = (uint32_t *) malloc(VEC_SIZE);
	if (tmpv) {
		memcpy(tmpv, x_var, VEC_SIZE);
		qsort(tmpv, n, sizeof(uint32_t), u32_cmp);
		#pragma omp parallel for simd private(i) reduction(+ : sz)
		for (i = 1; i < n; i++) {
			sz += (uint32_t) (tmpv[i] != tmpv[i - 1]);
		}
		tbl.tab_size = sz;
		tbl.count = (uint32_t *) calloc(sz, sizeof(uint32_t));
		tbl.category = (uint32_t *) calloc(sz, sizeof(uint32_t));
		if (tbl.category && tbl.count) {
			j = 0;
			tbl.count[j] = 1;
			tbl.category[j] = tmpv[j];
			for (i = 1; i < n; i++) {
				sz = (tmpv[i] == tmpv[i - 1]);
				tbl.count[j] += sz;
				j += !sz;
				tbl.count[j] += !sz;
				tbl.category[j] = tmpv[i];
			}
		}
	}
	free(tmpv);
	return tbl;
}

void free_uni_table(uni_table tbl) {
	free(tbl.count);
	free(tbl.category);
}

#ifdef DEBUG
int main() {
  uint32_t x[] = {28, 14, 15, 26, 11, 15, 9, 27, 13, 11, 12, 13, 15};
  uint32_t const n = 13;
  uint32_t i;
  uni_table mytbl = unitabl(x, n);
  printf("There are %u categorical values\n", mytbl.tab_size);
  for (i = 0; i < mytbl.tab_size; i++) {
    printf("Count of category \"%u\": %u\n", mytbl.category[i], mytbl.count[i]);
  }
  free_uni_table(mytbl);
  return 0;
}
#endif

