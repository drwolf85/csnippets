#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <math.h>
#include <omp.h>

typedef struct dict{
    void *p;
    size_t n;
} dict;

bool cmp_dta(void *lh, void *rh, size_t n) {
    bool res = (bool) ((*(u_int8_t *) lh ) ^ (*(u_int8_t *) rh));
    for (size_t i = 1; i < n; i++) res |= (bool) ((*(u_int8_t *) (lh + i)) ^ (*(u_int8_t *) (rh + i)));
    return res;
}

bool not_in(void *ptr, size_t plen, dict *x, size_t n) {
    bool res = true;
    size_t i;
    #pragma omp parallel for private(i) reduction(& : res)
    for (i = 0; i < n; i++) {
        if (x[i].n == plen && res) {
            #pragma atomic update
            {
                res &= cmp_dta(ptr, x[i].p, plen);
            }
        }
    }
    return res;
}

double jaccard(dict *A, size_t a, dict *B, size_t b) {
    size_t i, cnt = 0;
    double res = nan("");
    if (a < 1 && b < 1) return res;
    for (i = 0; i < a; i++) {
        cnt += (size_t) !not_in(A[i].p, A[i].n, B, b);
    }
    res = (double) cnt;
    res /= (double) (a + b - cnt);
    return res;
}

size_t lzDict(dict *x, void *a, size_t n) {
    size_t tmp;
    size_t cnt = 0;
    size_t start = 0;
    size_t end = 1;
    if (x && a && n > 0) {
        while(end < n) {
            tmp = end - start;
            if (not_in(a + start, tmp, x, cnt)) { /* (`a[start:end]` not in `x`) */
                x[cnt].p = a + start;
                x[cnt].n = tmp;
                start = end;
                cnt++;
            }
            end++;
        }
    }
    return cnt;
}

double lzjd(void *pta, size_t __mema, size_t __size_a, void *ptb, size_t __memb, size_t __size_b) {
    size_t sza = __mema *__size_a;
    size_t szb = __memb *__size_b;
    size_t sda, sdb;
    double res = nan("");
    if (pta && ptb && sza && szb) {
        sda = sdb = 0;
        dict *A = calloc(sza, sizeof(dict));
        dict *B = calloc(szb, sizeof(dict));
        sda = lzDict(A, pta, sza);
        sdb = lzDict(B, ptb, szb);
        A = (dict *) realloc(A, sza);
        B = (dict *) realloc(B, szb);
        res = 1.0 - jaccard(A, sda, B, sdb);
        free(A);
        free(B);
    }
    return res;
}

/* Test function */
#include <string.h>
#include <stdio.h>

int main() {
    char x[] = "I have had a conversation with your talented friend.";
    char y[] = "I have had a conversation with your talented friend.";
    char z[] = "No amount of wisdom, as I said before, ever banishes these things...";
    size_t nx = strlen(x);
    size_t nz = strlen(z);
    double dst = lzjd(x, nx, sizeof(char), y, nx, sizeof(char));
    printf("Approximation of the Kolmogorov complexity using the `lzjd` algorithm:\n");
    printf("\tDistance between:\n\t- %s\n\t- %s\n\tis %f\n", x, y, dst);
    dst = lzjd(x, nx, sizeof(char), z, nz, sizeof(char));
    printf("\tDistance between:\n\t- %s\n\t- %s\n\tis %f\n", x, z, dst);
    return 0;
}
