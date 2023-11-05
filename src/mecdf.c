#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct values {
    double x;
    size_t i;
} values;

/**
 * @brief Comparison function
 * 
 * @param aa void pointer to a `values` structure
 * @param bb void pointer to another `values` structure
 * @return int 
 */
int cmp_vals(void const *aa, void const *bb) {
    values a = *(values *) aa;
    values b = *(values *) bb;
    int res = 2 * (a.x > b.x) - 1;
    if (!res) res = 2 * (a.i > b.i) - 1;
    return res;
}

/**
 * @brief Binary search
 * 
 * @param x Benchmark value (double)
 * @param v Presorted vector of `values` structures of length `n`
 * @param n Length of the vector of `values`
 * @return size_t 
 */
long long binary_search(double x, values *v, size_t n) {
    long long  tmp, low, mid, high, res = -1;
    low = 0;
    high = n - 1;
    while (low < high) {
        mid = high + low;
        mid >>= 1;
        // printf("%lld %lld %lld %f\n", low, mid, high, v[mid].x);
        tmp = (long long) (x <= v[mid].x);
        high = (mid - 1) * tmp + (1 - tmp) * high;
        low = (mid + 1) * (1 - tmp) + low * tmp;
    }
    tmp = (long long) (x < v[0].x);
    res = (1 - tmp) * low - tmp;
    return res;
}

/**
 * @brief Multivariate empirical cumulative distribution function
 * 
 * @param x Pointer to a vector of values in input
 * @param dta Pointer to a matrix dataset of reference (column-major format)
 * @param n Number of records in the dataset
 * @param p Number of variables in the dataset
 * @return double 
 */
double mecdf(double *x, double *dta, size_t n, size_t p) {
    values v[p][n];
    char tmp;
    long long mnpos, pos[p];
    size_t i, j, whv;
    double res = 0.0;
    /* Copy the data in a structure to sort */ /** 
     *      NOTE: do this step outside the function to
     *            enhance performances! */
    #pragma omp parallel for private(i, j) collapse(2)
    for (j = 0; j < p; j++) {
        for (i = 0; i < n; i++) {
            v[j][i].x = dta[n * j + i];
            v[j][i].i = i;
        }
    }
    /* Sort with indices and search */
    #pragma omp parallel for private(j)
    for (j = 0; j < p; j++) {
        /** NOTE: Also the sorting should happen 
         *        outside the function to enhance 
         *        performances */
        qsort(v[j], n, sizeof(values), cmp_vals);
    }
    // for (j = 0; j < p; j++) {
    //     for (i = 0; i < n; i++) {
    //         printf("%f [%lu] - ", v[j][i].x, v[j][i].i);
    //     }
    //     printf("\n\n");
    // }
    /* Sort with indices and search */
    for (j = 0; j < p; j++) {
        /** NOTE: Also the sorting should happen 
         *        outside the function to enhance 
         *        performances */
        qsort(v[j], n, sizeof(values), cmp_vals);
        /* If the function receives the sorted structure as an input
           The binary search is the first step to perform */
        pos[j] = binary_search(x[j], v[j], n);
    }
    /* Find the minimum pos and variable of reference */
    mnpos = *pos;
    whv = 0;
    for (j = 1; j < p; j++) {
        whv += (mnpos > pos[j]) * (j - whv);
        mnpos += (mnpos > pos[j]) * (pos[j] - mnpos);
    }
    /* Estimation of the function */
    for (i = 0; (long long) i <= mnpos; i++) {
        tmp = 1;
        for (j = 0; j < p; j++) {
            tmp &= (dta[j * n + v[whv][i].i] <= x[j]);
        }
        res += (double) tmp;
    }
    return res / n;
}

/* Test function */
#define N 50
int main() {
    size_t i;
    double x[] = {0.165606409587784, 1.02140720582657, -0.123251973748602, -1.80563371563203, 0.0386651712520566, -0.84714066198676, 1.60820877821724, -0.23524530100488, -0.415438944288957, -0.0686008762861995, 0.0605161809320745, -0.57218469244946, 1.74830549897781, 0.455599834143699, -1.08741958061828, 0.790176764540126, -1.03017423700734, -0.685727401740929, 0.573624771205902, 0.470935127115954, -0.472059714526737, 0.388026422251463, -1.07739680185529, 0.190990633675878, 0.660474214133185, -1.16005584190488, 1.23465604259005, 0.597687455157315, -0.174303897439662, -0.0686281078031753, 0.38929918178785, 0.422362013840637, 0.831424664435272, -0.44145270294643, 0.176664392306577, 0.0551217753763783, -0.0865528234767478, 0.66238473027904, -0.210816261774775, -0.0115478158543238, -1.15330791929347, -0.27084812536643, -0.590031366931354, 0.507596472069195, -0.141013764598701, 0.260608960674256, -0.478740758871529, 1.55533963590707, 1.20711262112572, 0.69319944404685, 0.385280381335301, 0.826741412365128, 0.230618538158784, -0.804732790093561, 0.893756408570636, -1.59051673249855, -0.379368836367661, 0.98429272571221, 1.58392057179799, -0.399034445572324, 0.491446525054272, 0.199291299734876, 0.171709812753422, -0.601830971188744, -0.410479194731784, 0.869994551875666, 0.71037862364079, 0.851293616813694, 0.390972440708983, -0.400772827947995, -0.138536461377202, 0.468329823414287, 0.317801488115968, -1.17052660513811, 0.225953017304963, -0.124406414247572, -1.30231081559134, -1.18786072000791, 1.265656869665, 0.108880807057481, -0.875514451321625, 0.00319494768350267, 2.01328787123604, -0.251246965645935, -0.309957315910894, -0.481886373724893, 0.973634045496021, 0.137283451367027, -0.595368715196061, -1.12934664706978, 0.287168698337181, -1.29649216212074, -0.0179901068989467, -0.338477916349504, 0.88278178909952, -0.338589355839148, -3.3740091950637, 0.2714247519618, 2.12668246739218, 1.83120715826681};
    double y[] = {0.0, 0.0};
    values v[N] = {0};
    double res;

    for (i = 0; i < N; i++) {
        v[i].x = x[i];
        v[i].i = i;
    }
    qsort(v, N, sizeof(values), cmp_vals);
    for (i = 0; i < N; i++) {
        printf("(%lu : %f [%lu]) ", i, v[i].x, v[i].i);
    }
    printf("\n");
    printf("Position of -2: %lld\n", binary_search(-2.0, v, N));
    printf("Position of 0: %lld\n", binary_search(0.0, v, N));
    
    res = mecdf(y, x, N, 2);
    printf("ECDF(0.0, 0.0) = %f\n", res);
    
    return 0;
}
