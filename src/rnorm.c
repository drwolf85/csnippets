#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/** 
 * The function rnorm() is a C function that generates a random number from a normal distribution with
 * mean mu and standard deviation sd
 * 
 * @param mu mean of the normal distribution
 * @param sd standard deviation
 * 
 * @return A random number from a normal distribution with mean mu and standard deviation sd.
 */
double rnorm(double mu, double sd) {
   unsigned long u, v, m = (1 << 16) - 1;
   double a, b, s;
   u = rand();
   v = (((u >> 16) & m) | ((u & m) << 16));
   m = ~(1 << 31);
   u &= m;
   v &= m;
   a = ldexp((double) u, -30) - 1.0;
   s = a * a;
   b = ldexp((double) v, -30) - 1.0;
   s += b * b * (1.0 - s);
   s = -2.0 * log(s) / s;
   a = b * sqrtf(s);
   return mu + sd * a;
}

/* Main function to test the random generation of a normal variable */
int main() {
   double tmp;
   srand(time(NULL)); /* Initialize the random generator */

   for (int i = 1; i <= 40; i++) {
       tmp = rnorm(0.0, 1.0);
       if (tmp >= 0.0) printf(" ");
       printf("%f\t", tmp);
       if (i % 5 == 0) printf("\n");
   }
   return 0;
}
    
