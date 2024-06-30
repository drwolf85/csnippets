#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

/**
 * The function calculates the Hamming distance between two strings of characters using parallel
 * processing.
 * 
 * @param x The parameter x is a pointer to a character array representing the first string for which
 * we want to calculate the Hamming distance.
 * @param y The parameter "y" is a pointer to a character array, which is used as input to the
 * hamming_distance function. It is likely that this array represents a binary string or bit sequence.
 * @param n The parameter `n` represents the length of the input strings `x` and `y`.
 * 
 * @return The function `hamming_distance` returns the Hamming distance between two strings `x` and `y`
 * of length `n`. The Hamming distance is the number of positions at which the corresponding symbols
 * are different in the two strings.
 */
uint64_t hamming_distance(char *x, char *y, size_t n) {
    size_t i;
    char tmp, k;
    uint64_t res = 0;
    #pragma omp parallel for private(i, tmp, k) reduction(+ : res)
    for (i = 0; i < n; i++) {
        tmp = x[i] ^ y[i];
        for (k = 0; k < 8; k++)
            res += (uint64_t) ((tmp >> k) & 1);
    }
    return res;
}

/* Test function */
int main() {
    char x[] = "This is a test!";
    char y[] = "This is a nest.";
    char z[] = "That is a rest.";
    uint64_t hd = hamming_distance(x, y, 15);
    double res = (double) hd;
    printf("Computed Hamming distance is %lu\n", hd);
    res /= 0.5 * (res + (double) hamming_distance(x, z, 15) + (double) hamming_distance(z, y, 15));
    printf("Computed Steinhaus transform of the Hamming distance is %f\n", res);
    return 0;
}
