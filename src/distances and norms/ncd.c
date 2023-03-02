/**
 * @file ncd.c
 * @brief  Normalized Compression Distance
 */
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zconf.h>
#include <zlib.h>
#include <stdbool.h>
#include <lzma.h>

double ncd_z(void *x, size_t __mem_x, size_t __size_x, 
           void *y, size_t __mem_y, size_t __size_y) {
    double res = nan("");
    void *xy, *zx, *zy, *zxy;
    int x_res, y_res, xy_res;
    /* Compute data lengths */
    size_t const x_len = __mem_x * __size_x; 
    size_t const y_len = __mem_y * __size_y;
    size_t const xy_len = x_len + y_len;
    size_t x_n_dst = compressBound(x_len);
    size_t y_n_dst = compressBound(y_len);
    size_t xy_n_dst = compressBound(xy_len);
    /* Allocate memory */
    xy = malloc(xy_len);
    zx = malloc(x_n_dst);
    zy = malloc(y_n_dst);
    zxy = malloc(xy_n_dst);
    if (x && y && xy && zx && zy && zxy) {
        /* Concatenate data */
        memcpy(xy, x, x_len);
        memcpy(xy + x_len, y, y_len);
        /* Compressing data */
        x_res = compress2((Bytef *) zx, &x_n_dst, (Bytef *) x, x_len, 9);
        y_res = compress2((Bytef *) zy, &y_n_dst, (Bytef *) y, y_len, 9);
        xy_res = compress2((Bytef *) zxy, &xy_n_dst, (Bytef *) xy, xy_len, 9);
        if (x_res == Z_OK && y_res == Z_OK && xy_res == Z_OK) {
            res = (double) xy_n_dst;
            xy_n_dst = (x_n_dst > y_n_dst) * y_n_dst + (x_n_dst <= y_n_dst) * x_n_dst;
            y_n_dst += (x_n_dst > y_n_dst) * (x_n_dst - y_n_dst);
            x_n_dst = xy_n_dst;
            res -= (double) x_n_dst;
            res /= (double) y_n_dst;
        }
    }
    free(xy);
    free(zx);
    free(zy);
    free(zxy);
    return res;
}

double ncd_lzma(void *x, size_t __mem_x, size_t __size_x, 
          void *y, size_t __mem_y, size_t __size_y) {
    size_t const x_len = __mem_x * __size_x; 
    size_t const y_len = __mem_y * __size_y;
    size_t const xy_len = x_len + y_len;
    void *zx = malloc(x_len);
    void *zy = malloc(y_len);
    void *zxy = malloc(xy_len);
     void *xy = malloc(xy_len);
    uint32_t preset = 9; /* Compression level */
    lzma_check check = LZMA_CHECK_CRC64;
    lzma_stream x_strm = LZMA_STREAM_INIT;
    lzma_stream y_strm = LZMA_STREAM_INIT;
    lzma_stream xy_strm = LZMA_STREAM_INIT;
    lzma_action action;
    lzma_ret ret;
    size_t x_n_dst = 0;
    size_t y_n_dst = 0;
    size_t xy_n_dst = 0;
    double res = nan("");

    if (xy && zx && zy && zxy) {
        /* Concatenate data */
        memcpy(xy, x, x_len);
        memcpy(xy + x_len, y, y_len);
        /* Compressing data in x */
        ret = lzma_easy_encoder(&x_strm, preset, check);
        x_strm.next_in = (uint8_t *) x;
        x_strm.avail_in = x_len;
        x_strm.next_out = (uint8_t *) zx;
        x_strm.avail_out = x_len;
        do {
            action = LZMA_RUN;
            ret = lzma_code(&x_strm, action);
            if (ret == LZMA_OK) {
                x_n_dst += x_len - x_strm.avail_out;
            }
        } while(x_strm.avail_out == 0);
        action = LZMA_FINISH;
        ret = lzma_code(&x_strm, action);
        x_n_dst += x_len - x_strm.avail_out;
        lzma_end(&x_strm);
        /* Compressing data in y */
        ret = lzma_easy_encoder(&y_strm, preset, check);
        y_strm.next_in = (uint8_t *) y;
        y_strm.avail_in = y_len;
        y_strm.next_out = (uint8_t *) zy;
        y_strm.avail_out = y_len;
        do {
            action = LZMA_RUN;
            ret = lzma_code(&y_strm, action);
            if (ret == LZMA_OK) {
                y_n_dst += y_len - y_strm.avail_out;
            }
        } while(y_strm.avail_out == 0);
        action = LZMA_FINISH;
        ret = lzma_code(&y_strm, action);
        y_n_dst += y_len - y_strm.avail_out;
        lzma_end(&y_strm);
        /* Compressing data in xy */
        ret = lzma_easy_encoder(&xy_strm, preset, check);
        xy_strm.next_in = (uint8_t *) xy;
        xy_strm.avail_in = xy_len;
        xy_strm.next_out = (uint8_t *) zxy;
        xy_strm.avail_out = xy_len;
        do {
            action = LZMA_RUN;
            ret = lzma_code(&xy_strm, action);
            if (ret == LZMA_OK) {
                xy_n_dst += xy_len - xy_strm.avail_out;
            }
        } while(xy_strm.avail_out == 0);
        action = LZMA_FINISH;
        ret = lzma_code(&xy_strm, action);
        xy_n_dst += xy_len - xy_strm.avail_out;
        lzma_end(&xy_strm);
        res = (double) xy_n_dst;
        xy_n_dst = (x_n_dst > y_n_dst) * y_n_dst + (x_n_dst <= y_n_dst) * x_n_dst;
        y_n_dst += (x_n_dst > y_n_dst) * (x_n_dst - y_n_dst);
        x_n_dst = xy_n_dst;
        res -= (double) x_n_dst;
        res /= (double) y_n_dst;
    }
    free(xy);
    free(zx);
    free(zy);
    free(zxy);
    return res;
}

/* Test function 
Compile with gcc `ncd.c -Os -march=native -lm -lz -llzma`
*/
int main() {
    char x[] = "I have had a conversation with your talented friend.";
    char y[] = "I have had a conversation with your talented friend.";
    char z[] = "No amount of wisdom, as I said before, ever banishes these things...";
    size_t nx = strlen(x);
    size_t ny = strlen(y);
    double dst = ncd_z(x, nx, sizeof(char), y, ny, sizeof(char));
    printf("Approximation of the Kolmogorov complexity using `libz`:\n");
    printf("\tDistance between:\n\t- %s\n\t- %s\n\tis %f\n", x, y, dst);
    dst = ncd_z(x, nx, sizeof(char), z, nx, sizeof(char));
    printf("\tDistance between:\n\t- %s\n\t- %s\n\tis %f\n", x, z, dst);

    #include "../.data/iris.h"
    nx = N * P;
    dst = ncd_z(x_iris, nx, sizeof(double), x_iris, nx, sizeof(double));
    printf("\tDistance between (iris, iris) is %f\n", dst);
    printf("---\nApproximation of the Kolmogorov complexity using `liblzma`:\n");
    dst = ncd_lzma(x_iris, nx, sizeof(double), x_iris, nx, sizeof(double));
    printf("\tDistance between (iris, iris) is %f\n", dst);
    return 0;
}
