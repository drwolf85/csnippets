#include <stdio.h>

#define N 8

int main() {
	int i, j;
	printf("Skipping when i == j.\n");

	for (j = 0; j < N; j++) {
		printf("j = %d\t", j);
		for (i = 0; i < N; i++) {
			if (i != j) printf("%d ", i);
		}
		printf("\n");
	}
	return 0;
}
