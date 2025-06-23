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

static inline size_t arrange(void *arr, size_t n, size_t sz, void *ref, int (*cmp)(void const *, void const *, void const *)) {
        size_t j, end = n - 1;
        int64_t i = -1;
        int32_t tmp;
        uint8_t *ptr = (uint8_t *) arr;

        for (j = 0; j < n - 1; j++) {
                tmp = (int32_t) cmp((void *) &ptr[j * sz], (void *) &ptr[end * sz], ref);
                if (tmp < 0) {
                        i++;
                        exchange((void *) &ptr[sz * (size_t) i], (void *) &ptr[j * sz], sz);
                }
        }
        exchange((void *) &ptr[sz * (size_t) (i + 1)], (void *) &ptr[end * sz], sz);
        return (size_t) (i + 1);
}

extern void qsort3(void *arr, size_t n, size_t sz, void *ref, int (*cmp3)(void const *, void const *, void const *)) {
	size_t mdp;
        uint8_t *a;
	if (arr && ref && cmp3 && n > 1 && sz > 0) {
		a = (uint8_t *) arr;
		mdp = arrange(a, n, sz, ref, cmp3);
		if (mdp > 1) qsort3((void *) a, mdp, sz, ref, cmp3);
		if (mdp < n - 1) qsort3((void *) &a[(mdp + 1) * sz], n - mdp - 1, sz, ref, cmp3);
	}
}

#ifdef DEBUG

#include <math.h>

int mycmp(void const *aa, void const *bb, void const *pp) {
	double a = *(double *) aa;
	double b = *(double *) bb;
	double p = *(double *) pp;
	int res = 2 * (int) (fabs(a - p) > fabs(b - p)) - 1;
	return res;
}

#define N 6

int main(void) {
	double a[N] = {0};
	double piv = 0.0;
	size_t i;
	for (i = 0; i < N; i++) a[i] = ldexp((double) arc4random(), -30) - 2.0;
	printf("Original: ");
	for (i = 0; i < N; i++) printf("%f ", a[i]);
	printf("\n");
	qsort3((void *) a, N, sizeof(double), (void *) &piv, mycmp);
	printf("  Sorted: ");
	for (i = 0; i < N; i++) printf("%f ", a[i]);
	printf("\n");
	return 0;
}

#endif

