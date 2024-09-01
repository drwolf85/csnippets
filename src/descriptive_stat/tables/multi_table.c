#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../data structures/multi_tab.h"

static uint32_t p;

int u32v_cmp(const void *aa, const void *bb) {
  uint32_t i;
  uint32_t res = 0;
  uint32_t *a = (uint32_t *) aa;
  uint32_t *b = (uint32_t *) bb;
  for (i = 0; i < p && res == 0; i++) {
    res += (a[i] > b[i]);
  }
  return 2 * ((int) res) - 1;
}

/* x_mat is a pointer to a matrix stored in a row_major format */
multi_tab mlttabl (uint32_t *x_mat, uint32_t n) {
	uint32_t i, j, k, tof, sz = 1;
	multi_tab tbl;
	uint32_t *tmpv = (uint32_t *) malloc(n * p * sizeof(uint32_t));
	if (tmpv) {
		memcpy(tmpv, x_mat, n * p * sizeof(uint32_t));
		qsort(tmpv, n, p * sizeof(uint32_t), u32v_cmp);
    /* for (i = 0, k = 0; i < n; i++) for (j = 0; j < p; j++, k++) printf("%u%s", tmpv[k], j == (p - 1) ? "\n" : " "); */
		#pragma omp parallel for simd private(i, j, tof) reduction(+ : sz)
		for (i = 1; i < n; i++) {
		  tof = 0;
		  for (j = 0; j < p; j++) {
		    tof += (uint32_t) (tmpv[p * i + j] != tmpv[p * (i - 1) + j]);
		  }
			sz += (tof > 0);
		}
		tbl.n_vars = p;
		tbl.tab_size = sz;
    /* printf("Size: %u\n", sz); */
		tbl.counts = (uint32_t *) calloc(sz, sizeof(uint32_t));
		tbl.categories = (uint32_t **) calloc(sz, sizeof(uint32_t *));
		tof = 0;
		#pragma omp parallel for simd private(j) reduction(+ : tof)
		for (j = 0; j < sz; j++) {
  		tbl.categories[j] = (uint32_t *) calloc(p, sizeof(uint32_t));
  		tof += (tbl.categories[j] == NULL);
		}
		if (tbl.categories && tbl.counts && tof == 0) {
			j = 0;
			tbl.counts[j] = 1;
			for (k = 0; k < p; k++) tbl.categories[j][k] = tmpv[k];
			for (i = 1; i < n; i++) {
			  tof = 1;
		    for (k = 0; k < p; k++) {
		      tof &= (uint32_t) (tmpv[p * i + k] == tmpv[p * (i - 1) + k]);
		    }
				tbl.counts[j] += tof;
				if (!tof) {
				  j++;
				  tbl.counts[j]++;
			    for (k = 0; k < p; k++) tbl.categories[j][k] = tmpv[p * i + k];
				}
			}
		}
	}
	free(tmpv);
	return tbl;
}

void free_multi_tab(multi_tab tbl) {
  uint32_t j;
  for (j = 0; j < tbl.tab_size; j++) {
    free(tbl.categories[j]);
  }
	free(tbl.categories);
	free(tbl.counts);
	tbl.tab_size = 0;
	tbl.n_vars = 0;
}

#ifdef DEBUG
int main() {
  uint32_t x[] = {2, 4, \
  		            1, 6, \
  		            1, 5, \
  		            9, 7, \
  		            1, 6, \
  		            1, 5, \
  		            1, 5};
  uint32_t const n = 7;
  uint32_t i, j;
  p = 2;
  multi_tab mytbl = mlttabl(x, n);
  printf("There are %u unique sets of categorical values\n", mytbl.tab_size);
  for (i = 0; i < mytbl.tab_size; i++) {
    printf("Count of category");
    for (j = 0; j < mytbl.n_vars; j++) {
      printf(" \"%u\"", mytbl.categories[i][j]);
    }
    printf(": %u\n", mytbl.counts[i]);
  }
  free_multi_tab(mytbl);
  return 0;
}
#endif

