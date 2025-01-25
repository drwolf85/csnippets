#ifndef SPARSE_MAT_ARR_H
#define SPARSE_MAT_ARR_H

#define COLUMN_MAJOR 0
#define ROW_MAJOR 1
#define DIAGONAL 2
#define ANTI_DIAGONAL 3

typedef struct sparse_matrix_array {
	double *m;
	unsigned nr;
	unsigned nc;
	unsigned char type;
	unsigned *idx;
	unsigned *pos;
} sp_mat;

#endif

