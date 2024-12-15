#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

uint64_t pack_bits(uint8_t *x, uint8_t n) {
    uint64_t res = 0;
    uint8_t i;
    if (n <= 64) for (i = 0; i < n; i++) 
        res |= ((uint64_t) x[i] & 1ULL) << (uint64_t) i;
    return res;
}

uint64_t pack_bits_dbl(double *x, uint8_t n) {
    uint64_t res = 0;
    uint8_t i;
    if (n <= 64) for (i = 0; i < n; i++) 
        res |= (uint64_t) (x[i] >= 0.5) << (uint64_t) i;
    return res;
}

#ifdef DEBUG
int main() {
    uint8_t bits[8] = {0, 1, 1, 0, 1, 0, 1, 0};
    double dbits[8] = {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    printf("Test packed bits\n\tfrom `uint8_t`\t%llu\n", pack_bits(bits, 8));
    printf("\tfrom `double`\t%llu\n", pack_bits_dbl(dbits, 8));
    return 0;
}
#endif
