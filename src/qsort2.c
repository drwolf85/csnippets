#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static inline void exchange(void *a, void *b, size_t sz) {
	size_t i;
	uint8_t u;
	uint8_t *ua, *ub;
	if (a == b) return;
	ua = (uint8_t *) a;
	ub = (uint8_t *) b;
	for (i = 0; i < sz; i++) {
		u = ua[i];
		ua[i] = ub[i];
		ub[i] = u;
	}
}

static inline size_t arrange(void *arr, size_t n, size_t sz, int (*cmp)(void const *, void const *)) {
	size_t j, end = n - 1;
	int64_t i = -1;
	int32_t tmp;
	uint8_t *ptr = (uint8_t *) arr;
	for (j = 0; j < n - 1; j++) {
		tmp = (int32_t) cmp((void *) &ptr[j * sz], (void *) &ptr[end * sz]);
		if (tmp < 0) {
			i++;
			exchange((void *) &ptr[sz * (size_t) i], (void *) &ptr[j * sz], sz);
		}
	}
	exchange((void *) &ptr[sz * (size_t) (i + 1)], (void *) &ptr[end * sz], sz);
	return (size_t) (i + 1);
}

extern void qsort2(void *arr, size_t n, size_t sz, int (*cmp)(void const *, void const *)) {
	size_t mdp;
	uint8_t *a;
	if (arr && cmp && sz > 0 && n > 1) {
		a = (uint8_t *) arr;
		mdp = arrange(a, n, sz, cmp);
		if (mdp > 1) qsort2((void *) a, mdp, sz, cmp);
		if (mdp < n - 1) qsort2((void *) &a[(mdp + 1) * sz], n - mdp - 1, sz, cmp);
	}
}


#ifdef DEBUG

#include <math.h>

int mycmp(void const *aa, void const *bb) {
        double a = *(double *) aa;
        double b = *(double *) bb;
        int res = 2 * (int) (fabs(a) > fabs(b)) - 1;
        return res;
}

#define N 6

int main(void) {
        double a[N] = {0};
        size_t i;
        for (i = 0; i < N; i++) a[i] = ldexp((double) arc4random(), -30) - 2.0;
        printf("Original: ");
        for (i = 0; i < N; i++) printf("%f ", a[i]);
        printf("\n");
        qsort2((void *) a, N, sizeof(double), mycmp);
        printf("  Sorted: ");
        for (i = 0; i < N; i++) printf("%f ", a[i]);
        printf("\n");
        return 0;
}

#endif

