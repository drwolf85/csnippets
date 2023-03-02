#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/**
 * @brief C code implementing Hopfield networks (Hopfield J., 1982)
 */

#define MAX_ITERATIONS 100

/**
 * The "params" type is a struct that contains two double pointers and a size_t variable.
 * @property {double} W - W is a pointer to a double, which represents the weights of a neural
 * network. The weights are typically stored in a matrix or array format, and this pointer
 * points to the beginning of that matrix or array.
 * @property {double} b - The "b" property is a pointer to a double, which is used to store bias
 * values in a machine learning model. The "params" struct seems to be defining a set of parameters for
 * a model, including weights ("W"), biases ("b"), and the size of the parameter set ("
 * @property {size_t} sz - `sz` is a variable of type `size_t` that represents the size of the arrays
 * `W` and `b` in the `params` struct. It is used to keep track of the number of weights and
 * biases in a neural network model.
 */
typedef struct {
    double *W;
    double *b;
    size_t sz;
} params;

/**
 * The function `net_alloc` allocates memory for a neural network's parameters.
 * 
 * @param dim The parameter `dim` represents the dimension of the neural network layer for which the
 * memory is being allocated. The function `net_alloc` allocates memory for the weight matrix `W`, bias
 * vector `b`, and the size of the layer `sz` using the input dimension `dim`.
 * 
 * @return The function `net_alloc` is returning a pointer to a `params` struct.
 */
params * net_alloc(size_t dim) {
    params *par = malloc(sizeof(params));
    if (par) {
        par->W = malloc(dim * dim * sizeof(double));
        par->b = malloc(dim * sizeof(double));
        par->sz = dim;
    }
    return par;
}

/**
 * The function frees memory allocated for a given parameter struct.
 * 
 * @param par The parameter `par` is a pointer to a struct of type `params`. This function is
 * used to free the memory allocated for the parameters of a neural network model. The `params` struct
 * contains the weights (`W`) and biases (`b`) of the neural network, which are dynamically allocated.
 */
void net_free(params *par) {
    if (par) {
        free(par->W);
        free(par->b);
        free(par);
    }
}

/**
 * The function trains a neural network by adjusting weights based on patterns in the data and then
 * averaging over the weights.
 * 
 * @param par A pointer to a struct containing the parameters for the neural network.
 * @param dat A pointer to an array of uint64_t integers representing binary data patterns.
 * @param n The number of samples in the data.
 */
void train_net(params *par, uint64_t *dat, size_t n) {
    size_t s, i, j, k;
    int xi, xj;
    double invn = 1.0 / n;
    size_t d;
    if (par) if (par->W && par->b) {
        /* Initialize the parameters */
        d = par->sz;
        if (d > 64) d = 64;
        memset(par->W, 0, sizeof(double) * d * d);
        memset(par->b, 0, sizeof(double) * d);
        /* Adjusting weights based on the patterns in the data */
        #pragma omp parallel for private(s, i, j, xi, xj)
        for (s = 0; s < n; s++) { /* Loop over the samples */
            for (j = 0; j < d; j++) {
                xj = 2 * ((int) (dat[s] >> j) & 1) - 1;
                #pragma omp atomic
                par->b[j] += (double) xj;
                for (i = 0; i < d; i++) {
                    xi = 2 * ((int) (dat[s] >> i) & 1) - 1;
                    #pragma omp atomic 
                    par->W[j * d + i] += (double) (xi * xj);
                }
            }
        }
        /* Averaging over the weights */
        #pragma omp parallel for simd private(j)
        for (i = 0; i < d; i++) {
            for (j = 0; j <= i; j++) {
                par->W[j * d + i] *= (double) (i != j) * invn;
                par->W[i * d + j] = par->W[j * d + i];
            }
            par->b[i] *= invn;
        }
    }
}

/**
 * The function calculates the energy of a given data point using a set of parameters.
 * 
 * @param par A pointer to a struct containing the parameters of the energy function. It should have
 * two fields: "W", a pointer to a double array representing the weights of the connections between
 * neurons, and "b", a pointer to a double array representing the biases of the neurons.
 * @param dat dat is an input parameter of type uint64_t, which is an unsigned 64-bit integer. It is
 * used as a binary representation of a data point, where each bit represents a feature or attribute of
 * the data point. The function calculates the energy of this data point based on the given weight
 * matrix
 * 
 * @return a double value which represents the energy of a given set of parameters and data.
 */
double energy(params *par, uint64_t dat) {
    size_t i, j, d;
    int xi, xj;
    double res = 0.0;
    if (par) if (par->W && par->b) {
        d = par->sz;
        if (d > 64) d= 64;
        #pragma omp parallel for simd private(xi, j, xj) reduction(- : res)
        for (i = 0; i < d; i++) {
            xi = 2 * ((int) (dat >> i) & 1) - 1;
            res -= xi * par->b[i];
            for (j = 0; j < d; j++) {
                xj = 2 * ((int) (dat >> j) & 1) - 1;
                res -= 0.5 * (double) (xi * xj) * par->W[i * d + j] * (i != j);
            }
        }
    }
    return res;
}

/**
 * The function updates a given data point using a specified set of parameters and returns the updated
 * data point.
 * 
 * @param par A pointer to a struct containing parameters for the energy function and the weight matrix
 * and bias vector used in the data update function.
 * @param dat The input data that needs to be updated. It is of type uint64_t, which means it is a
 * 64-bit unsigned integer.
 * 
 * @return a 64-bit unsigned integer (uint64_t) value named "res".
 */
uint64_t data_update(params *par, uint64_t dat) {
    size_t i, j, d, count = 0;
    int xi, xj;
    double sum, olde, newe;
    uint64_t res;
    if (par) if (par->W && par->b) {
        d = par->sz;
        if (d > 64) d= 64;
        newe = energy(par, dat);
        for (;;) {
            olde = newe;
            sum = 0.0;
            res = 0;
            #pragma omp parallel for simd private(j, xj) reduction(+ : sum) reduction(| : res)
            for (i = 0; i < d; i++) {
                sum += par->b[i];
                for (j = 0; j < d; j++) {
                    xj = 2 * ((int) (dat >> j) & 1) - 1;
                    sum += (double) xj * par->W[i * d + j] * (i != j);
                }
                res |= (sum >= 0.0) << i;
            }

            /** WARNING: Use masking if some values are forced to be constant! ...*/
            res = (~0xff & dat) | (0xff & res); /** ... otherwise comment or remove this line!!! */
            
            newe = energy(par, res);
            count++;
            if (count >= MAX_ITERATIONS || olde <= newe) {
                // printf("Total iterations for convergence: %lu\n", count);
                res = dat;
                break;
            }
            else {
                dat = res;
            }
        }
    }
    return res;
}

/* Test function */

#define M_DI 16 /* Length of the binary pattern */
#define M_DP 4 /* Data points */
int main () {
    uint64_t data[] = { 0x0105, 0x3801, 0xff00, 0xa4d1 };
    uint64_t res = 0x0100;
    params *mypar;
    int i, j;
    mypar = net_alloc(M_DI);
    if (mypar) {
        train_net(mypar, data, M_DP);
        printf("Weights:\n");
        for (i = 0; i < M_DI; i++) {
            for (j = 0; j < M_DI; j++) {
                if (mypar->W[i * M_DI + j] >= 0) printf(" ");
                printf("%.2f ", mypar->W[i * M_DI + j]);
            }
            printf("\n");
        }
        printf("\nThresholds:\n");
        for (i = 0; i < M_DI; i++) {
            printf("%.2f ", mypar->b[i]);
        }
        printf("\n\nTesting Energy function:\n");
        for (i = 0; i < M_DP; i++)
            printf("\tEnergy data[%d]=%08lX: %f\n", i, data[i], energy(mypar, data[i]));

        printf("\nTesting data updates for pattern retrival:\n"
               "\tEnergy of %08lX: %f\n", res, energy(mypar, res));
        res = data_update(mypar, res);
        printf("\tEnergy of %08lX ", res);
        printf("(updated value): %f\n", energy(mypar, res));
    }    
    net_free(mypar);
    return 0;
}
