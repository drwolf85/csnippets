#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

double * onehot(int *x, uint32_t n) {
    double *res = NULL;
    uint32_t i, m;
    if (x) { /* Get maximum number of categories */
        m = *x;
        for (i = 1; i < n; i++) 
            m += (uint32_t) (m < x[i]) * (x[i] - m); 
        res = (double *) calloc(n * m, sizeof(double));
        if (res) {
            for (i = 0; i < n; i++) {
                res[n * x[i] + i] = 1.0;
            }
        }
    }
    return res;
}

#ifdef DEBUG
int main() {
    int i, j, x[9] = {2,4,3,0,4,7,5,6,1};
    double *mat = onehot(x, 9);
    if (mat) {
        printf("Testing one-hot encoding:\n");
        for (i = 0; i < 9; i++) {
            printf("x[i] = %d\t OH: ", x[i]);
            for (j = 0; j < 8; j++) printf("%.0f ", mat[9 * j + i]);
            printf("\n");
        }
    }
    free(mat);
    return 0;
}
#endif
