#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
/* Shannon's entropy */
double shannon_entropy_d(double *prob, size_t n) {
	double res = nan(""), tmp = 0.0;
	bool test = true;
	size_t i;
	/* Checking if the input array is a probability distribution */
	if (prob) {
		for (i = 0; i < n; i++) {
			test &&= (prob[i] >= 0.0) && (prob[i] <= 1.0);
			tmp += prob[i];
		}
		test &&= (tmp == 1.0);
		if (test) {
			res = 0.0;
			for (i = 0; i < n; i++) {
				if (prob[i] > 0.0) 
					res += prob[i] * log(prob[i]);
			}		
		}
	}
	return res;
}


