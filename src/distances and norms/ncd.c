/**
 * @file ncd.c
 * @brief  Normalized Compression Distance
 */
 #include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zconf.h>
#include <zlib.h>

double ncd(void *x, size_t __mem_x, size_t __size_x, 
           void *y, size_t __mem_y, size_t __size_y) {
    double res = nan("");
    void *xy, *zx, *zy, *zxy;
    int x_res, y_res, xy_res;
    /* Compute data lengths */
    size_t const x_len = __mem_x * __size_x; 
    size_t const y_len = __mem_y * __size_y;
    size_t const xy_len = x_len * y_len;
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

/* Test function */
int main() {
    char x[] = "I have had a conversation with your talented friend.";
    char y[] = "No amount of wisdom, as I said before, ever banishes these things...";
    size_t nx = strlen(x);
    size_t ny = strlen(y);
    double dst = ncd(x, nx, sizeof(char), y, ny, sizeof(char));
    printf("Distrance between:\n\t- %s\n\t- %s\nis %f\n", x, y, dst);
    return 0;
}
