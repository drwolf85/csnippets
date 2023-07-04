#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double log_binom(int n, int k) {
    double res = lgamma((double) (n + 1));
    res -= lgamma((double) (k + 1));
    res -= lgamma((double) (n - k + 1)); 
    return res;
}

double dhyper(int x, int m, int n, int k) {
    double z = nan("");
    if ((x >= 0 && x >= (k - n)) && (x <= k && x <= m)) {
        z = log_binom(m, x);
        z += log_binom(n, k - x);
        z -= log_binom(m + n, k);      
        z = exp(z);
    }
    return z;
}

double phyper(int x, int m, int n, int k) {
    int j, i = k - n;
    double z = nan("");
    i *= (i > 0);
    if (x >= i && (x <= k && x <= m)) {
        z = 0.0;
        for (j = 0; j < x; j++) {
            z += dhyper(i + j, m, n, k);
        }
    }
    return z;
}

double qhyper(double p, int m, int n, int k) {
    int j, i = k - n;
    double z = nan("");
    i *= (i > 0);
    if (p >= 0.0 && p < 1.0) {
        z = 0.0;
        for (j = 0; z < p; j++) {
            z += dhyper(i + j, m, n, k);
        }
        z = (double) (i + j - 1);
        if ((z > (double) k || z > (double) m)) z = nan("");
    }
    else if (p == 1.0) {
        z = (double) (k * (k < m) + m * (m >= k));
    }
    return z;
}

double rhyper(int m, int n, int k) {
    unsigned long u, v;
    double z;
    u = rand();
    v = ~(1 << 31);
    u &= v;
    z = qhyper(ldexp((double) u, -31), m, n, k);
    return z;
}

/* Test function */
int main() {
    int x = 2;
    double d, p, q;
    double tmp;
    srand(time(NULL)); /* Initialize the random generator */
    d = dhyper(x, 5, 2, 3);
    p = phyper(x, 5, 2, 3);
    q = qhyper(0.95, 5, 2, 3);
    printf("x = %d, d = %f, p = %f, q = %.0f\n", x, d, p, q);
    /* Main function to test the random generation of a Hypergeometric variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rhyper(5, 2, 3);
        printf("%.0f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    return 0;
}
