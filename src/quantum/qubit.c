#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#ifndef CMPLX
# define CMPLX(x, y) ((x) + (y) * I)
#endif

/**
 * The qubit type represents a complex value.
 * @property {complex} value - The `value` property is a pointer to a `complex` data type.
 */
typedef struct qubit {
    complex double *value;
} qubit;

/**
 * The function initializes a qubit by setting its value to a random zero ket.
 * 
 * @return a pointer to a qubit structure.
 */
qubit * qubit_init() {
    unsigned long u, m = ~(1 << 31);
    double tmp;
    qubit *q = (qubit *) calloc(1, sizeof(qubit));
    if (q) {
        q->value = (complex double *) calloc(2, sizeof(complex double));
        if (q->value) {
            u = rand() & m;
            tmp = ldexp((double) u, -30) * M_PI;
            /* Setting random zero ket */
            q->value[0] = CMPLX(sin(tmp), cos(tmp));
            q->value[1] = CMPLX(0.0, 0.0);
        }
    }
    return q;
}

/**
 * The function qubit_free frees the memory allocated for a qubit structure.
 * 
 * @param q The parameter "q" is a pointer to a qubit structure.
 */
void qubit_free(qubit *q) {
    free(q->value);
    free(q);
}

/**
 * The function `quantum_not` swaps the values of the first and second elements in the `value` array of
 * a given `qubit` structure.
 * 
 * @param q The parameter `q` is a pointer to a qubit structure.
 */
void quantum_not(qubit *q) {
    complex double a;
    if (q) if (q->value) {
        a = q->value[0];    
        q->value[0] = q->value[1];
        q->value[1] = a;
    }
}

/**
 * The function "pauli_identity" does not perform any operations and simply returns.
 * 
 * @param q The parameter "q" is a pointer to a qubit object.
 * 
 * @return Nothing is being returned. The return type of the function is void, which means it does not
 * return any value.
 */
void pauli_identity(qubit *q) {
    return;
}

/**
 * The function pauli_X performs a Pauli-X gate operation on a qubit if it has a value.
 * 
 * @param q The parameter "q" is a pointer to a qubit object.
 */
void pauli_X(qubit *q) {
    if (q) if (q->value) {
        quantum_not(q);
    }
}

/**
 * The function pauli_Y applies the Pauli Y gate to a qubit.
 * 
 * @param q The parameter `q` is a pointer to a `qubit` structure.
 */
void pauli_Y(qubit *q) {
    complex double a;
    if (q) if (q->value) {
        a = q->value[0] * CMPLX(0.0, 1.0);
        q->value[0] = q->value[1] * CMPLX(0.0, -1.0);
        q->value[1] = a;
    }
}

/**
 * The function pauli_Z applies the Pauli-Z gate to a qubit by negating the imaginary part of its
 * value.
 * 
 * @param q The parameter "q" is a pointer to a qubit structure.
 */
void pauli_Z(qubit *q) {
    if (q) if (q->value) {
        q->value[1] = -q->value[1];
    }
}

/**
 * The function `sqrt_not` performs a square root not of the input qubit.
 * 
 * @param q The parameter "q" is a pointer to a qubit structure.
 */
void sqrt_not(qubit *q) {
    complex double a, b;
    if (q) if (q->value) {
        a = CMPLX(0.5, 0.5) * q->value[0] + CMPLX(0.5, -0.5) * q->value[1];
        b = CMPLX(0.5, -0.5) * q->value[0] - CMPLX(0.5, 0.5) * q->value[1];
        q->value[0] = a;
        q->value[1] = b;
    }
}

/**
 * The function performs the Hadamard transformation on a qubit.
 * 
 * @param q The parameter `q` is a pointer to a `qubit` structure.
 */
void hadamard(qubit *q) {
    complex double a, b;
    double const m_isqrt2 = sqrt(0.5);
    if (q) if (q->value) {
        a = (q->value[0] + q->value[1]) * m_isqrt2;
        b = (q->value[0] - q->value[1]) * m_isqrt2;
        q->value[0] = a;
        q->value[1] = b;
    }
}

/**
 * The function `r_x` applies a rotation around the x-axis to a qubit.
 * 
 * @param q The parameter `q` is a pointer to a qubit structure.
 * @param alpha The parameter "alpha" in the given code represents the rotation angle in radians. It is
 * used to rotate the state of a qubit around the X-axis on the Bloch sphere.
 */
void r_x(qubit *q, double alpha) {
    complex double a, b;
    double const c = cos(alpha * 0.5);
    double const s = sin(alpha * 0.5);
    if (q) if (q->value) {
        a = c * q->value[0] - CMPLX(0.0, s) * q->value[1];
        b = CMPLX(0.0, -s) * q->value[0] + c * q->value[1];
        q->value[0] = a;
        q->value[1] = b;
    }
}

/**
 * The function `r_y` applies a rotation around the y-axis to a qubit, given an angle `alpha`.
 * 
 * @param q The parameter `q` is a pointer to a qubit structure.
 * @param alpha The parameter "alpha" in the given code represents the rotation angle in radians. It is
 * used to calculate the values of "c" and "s" which are the cosine and sine of half of the rotation
 * angle, respectively. These values are then used to perform a rotation operation on the qubit
 */
void r_y(qubit *q, double alpha) {
    complex double a, b;
    double const c = cos(alpha * 0.5);
    double const s = sin(alpha * 0.5);
    if (q) if (q->value) {
    a = c * q->value[0] - s * q->value[1];
    b = s * q->value[0] + c * q->value[1];
    q->value[0] = a;
    q->value[1] = b;
    }
}

/**
 * The function `r_z` applies a rotation around the Z-axis to a qubit with a given angle.
 * 
 * @param q A pointer to a qubit structure. The qubit structure contains a complex array called "value"
 * that represents the state of the qubit. The value[0] element represents the probability amplitude of
 * the |0⟩ state, and the value[1] element represents the probability amplitude of the |
 * @param alpha The parameter "alpha" is a real number that determines the rotation angle in the
 * Z-axis.
 */
void r_z(qubit *q, double alpha) {
    double const am = alpha * 0.5;
    if (q) if (q->value) {
        q->value[0] *= cexp(CMPLX(0.0, -am));
        q->value[1] *= cexp(CMPLX(0.0, am));
    }
}

/**
 * The function `phase_shift` applies a phase shift to a qubit by multiplying its values by a complex
 * exponential.
 * 
 * @param q The parameter `q` is a pointer to a `qubit` structure.
 * @param delta The parameter "delta" represents the phase shift angle in radians. It determines the
 * amount by which the phase of the qubit is shifted.
 */
void phase_shift(qubit *q, double delta) {
    complex double const eid = cexp(CMPLX(0.0, delta));
    if (q) if (q->value) {
        q->value[0] *= eid;
        q->value[1] *= eid;
    }
}

/**
 * The function "observe_qubit" returns a random binary value based on the probability of measuring a
 * qubit in the |1> state.
 * 
 * @param q The parameter `q` is a pointer to a qubit structure.
 * 
 * @return The function `observe_qubit` returns a character value.
 */
char observe_qubit(qubit *q) {
    unsigned long u, m = ~(1 << 31);
    double tmp;
    char res = 0;

    if (q) if (q->value) {
        tmp = cabs(q->value[1]);
        u = rand() & m;
        tmp *= tmp;
#ifdef _DEBUG
        printf("\nDEBUG tmp = %f\n", tmp);
#endif
        res = (char) (ldexp((double) u, -31) <= tmp);
    }

    return res;
}

/**
 * The function calculates the expectation value of a qubit.
 * 
 * @param q The parameter "q" is a pointer to a qubit object.
 * 
 * @return the square of the modulus of the second element of the qubit's value array.
 */
double qubit_expectation(qubit *q) {
    double res = cabs(q->value[1]);
    return res * res;
}

/* Test function */
int main() {
    int i;
    qubit *q;
    
    srand(time(NULL)); /* Initialize the random generator */
    q = qubit_init(); /* Initialize the qubit */

    printf("Theoretical quantum state initially stored in memory:\n");
    printf("(%f%s%fi, %f%s%fi)\n\n",
           creal(q->value[0]), 
           cimag(q->value[0]) >= 0.0 ? "+" :"", 
           cimag(q->value[0]), 
           creal(q->value[1]), 
           cimag(q->value[1]) >= 0.0 ? "+" :"",
           cimag(q->value[1]));

    /* Applying quantum gates to obtain final superposition */
    pauli_identity(q);
    hadamard(q);
    pauli_X(q);
    phase_shift(q, 3.0 * M_PI_4);
    pauli_Y(q);
    r_x(q, M_PI_2 * 3);
    pauli_Z(q);
    r_y(q, M_PI * 0.123);
    sqrt_not(q);
    r_z(q, -0.321 * M_PI_2);
    quantum_not(q);
    hadamard(q);

    printf("Final superposition stored in memory:\n");
    printf("(%f%s%fi, %f%s%fi)\n\n",
           creal(q->value[0]), 
           cimag(q->value[0]) >= 0.0 ? "+" :"", 
           cimag(q->value[0]), 
           creal(q->value[1]), 
           cimag(q->value[1]) >= 0.0 ? "+" :"",
           cimag(q->value[1]));

    printf("Theorical expectation of the observed qubit status: %f\n",
           qubit_expectation(q));

    printf("Observations generated by accessing the status of the qubit:\n");
    for (i = 1; i <= 40; i++) {
        printf("\t%d", observe_qubit(q));
        if (i % 5 == 0) printf("\n");
    }

    qubit_free(q);
    return 0;
}
