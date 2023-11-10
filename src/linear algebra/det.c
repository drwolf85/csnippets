/**
 * @brief Determinant of a 2x2 matrix
 * 
 * @param A Pointer to a 2x2 matrix of data
 * @return double 
 */
double det_2x2(double *A) {
    return A[0] * A[3] - A[1] * A[2];
}

/**
 * @brief Determinant of a 3x3 matrix
 * 
 * @param A Pointer to a 3x3 matrix of data
 * @return double 
 */
double det_3x3(double *A) {
    double res = A[0] * A[4] * A[8];
    res += A[3] * A[7] * A[2];
    res += A[6] * A[1] * A[5];
    res -= A[6] * A[4] * A[2];
    res -= A[0] * A[7] * A[5];
    res -= A[3] * A[1] * A[8];
    return res;
}
