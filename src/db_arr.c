#include <stdio.h>
#include <stdlib.h>

#define N 4

int main() {
	int i, j;
	float **a;
	/* Allocate array */
	a = (float **) calloc(N, sizeof(float *));
	for (i = 0; i < N; i++) {
		a[i] = (float *) calloc(N - 1, sizeof(float));
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < i; j++) a[i][j] = (float) j + 1.f;
		for (j = i + 1; j < N; j++) a[i][j - 1] = (float) j + 1.f;
		for (j = 0; j < N - 1; j++) printf("%f ", a[i][j]);
		printf("\n");
	}
	/* Free allocated array */
	for (i = 0; i < N; i++) {
		free(a[i]);
	}
	free(a);
	return 0;
}
