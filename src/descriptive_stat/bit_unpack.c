#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t * bit_unpack_64(uint64_t *x) {
    uint8_t * res = (uint8_t *) malloc(64 * sizeof(uint8_t));
    uint8_t i;
    if (res) for (i = 0; i < 64; i++)
        res[i] = 0 != (uint8_t) ((*x >> i) & 1ULL);
    return res;
}

uint8_t * bit_unpack_32(uint32_t *x) {
    uint8_t * res = (uint8_t *) malloc(32 * sizeof(uint8_t));
    uint8_t i;
    if (res) for (i = 0; i < 32; i++)
        res[i] = 0 != (uint8_t) ((*x >> i) & 1ULL);
    return res;
}

uint8_t * bit_unpack_16(uint16_t *x) {
    uint8_t * res = (uint8_t *) malloc(16 * sizeof(uint8_t));
    uint8_t i;
    if (res) for (i = 0; i < 16; i++)
        res[i] = 0 != (uint8_t) ((*x >> i) & 1ULL);
    return res;
}

uint8_t * bit_unpack_8(uint8_t *x) {
    uint8_t * res = (uint8_t *) malloc(8 * sizeof(uint8_t));
    uint8_t i;
    if (res) for (i = 0; i < 8; i++)
        res[i] = 0 != (uint8_t) ((*x >> i) & 1ULL);
    return res;
}

#ifdef DEBUG

#define LOOK_AT(MY_NUM_BITS) {\
    if (bits) { \
        printf("%02X: ", val); \
        for (i = (MY_NUM_BITS); i > 0; i--) printf(" %u", bits[i - 1]); \
        printf("\n"); \
    } \
    free(bits);
}

int main() {
    uint8_t i, val = 85, *bits;
    uint16_t val16 = (uint16_t) val;
    uint32_t val32 = (uint32_t) val;
    uint64_t val64 = (uint64_t) val;
    printf("Test unpacked bits (value: %u)\n", val);
    bits = bit_unpack_8(&val);
    LOOK_AT(8);
    bits = bit_unpack_16(&val16);
    LOOK_AT(16);
    bits = bit_unpack_32(&val32);
    LOOK_AT(32);
    bits = bit_unpack_64(&val64);
    LOOK_AT(64);
    return 0;
}
#endif
