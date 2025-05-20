#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EM25 1.388794386496402089636e-11
#define NEM1M25 0.9999999999861121091627

/**
 * The function `dnorm` returns the value of the normal density function at `x` with mean `m` and
 * standard deviation `s`
 * 
 * @param x the value we're evaluating the PDF at
 * @param m mean
 * @param s standard deviation
 * 
 * @return The probability of x given the mean and standard deviation.
 */
double dnorm(double x, double m, double s) {
    double z = x - m;
    z /= s; 
    z = exp(-0.5 * z * z);
    z /= s * sqrt(2.0 * M_PI);
    return z;
}

/**
 * The function `pnorm` returns the probability that a random variable from a normal distribution
 * with mean `m` and standard deviation `s` is less than or equal to `x`
 * 
 * @param x the value of the random variable
 * @param m mean
 * @param s standard deviation
 * 
 * @return The probability of a random variable being less than or equal to x.
 */
double pnorm(double x, double m, double s) {
    double z = x - m;
    z /= s * sqrt(2.0); 
    z = 0.5 + 0.5 * erf(z);
    return z;
}

/**
 * It takes a probability, a mean, and a standard deviation, and returns the value of the normal
 * distribution at that probability
 * 
 * @param p the probability of the event
 * @param m mean
 * @param s standard deviation
 * 
 * @return The quantile function of the normal distribution.
 */
double qnorm(double p, double m, double s) {
    double old;
    double z, sdv; /* Initial approximation */
    z = sqrt(- (p >= 0.5 ? log(p) : log1p(-p)));
    z *= 0.1661 * z - 2.25;
    z += 1.758;
    z *= (double) (p >= 0.5) * 2.0 - 1.0;
    sdv = dnorm(z, 0.0, 1.0);
    /* int count = 0; */
    do {
        old = z;
        z += (p - pnorm(z, 0.0, 1.0)) / sdv;
        sdv = dnorm(z, 0.0, 1.0);
        /* count++; */
    } while (sdv > 1e-16 && fabs(old - z) > 1e-12);
    /* printf("count = %d\n", count); */
    return s * z + m;
}

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
   u = arc4random();
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

/** NOTE: The following functions are used for high-accuracy approximation of the qnorm function */
/* First rational Chebyshev approximation $R_{1(5,4)}(z)$ */
static inline double R1(double z) {
    double num = -7.784894002430293e-3 * z;
    double den =  7.784695709041462e-3 * z;
    num = (num - 3.223964580411365e-1) * z;
    num = (num - 2.400758277161838) * z;
    num = (num - 2.549732539343734) * z;
    num = (num + 4.374664141464968) * z;
    num += 2.938163982698783;
    den = (den + 3.224671290700398e-1) * z;
    den = (den + 2.445134137142996) * z;
    den = (den + 3.754408661907416) * z;
    den += 1.0;
    return num / den;
}

/* Second rational Chebyshev approximation $R_{2(5,5)}(z)$ */
static inline double R2(double z) {
    double num = -3.969683028665376e1 * z;
    double den = -5.447609879822406e1 * z;
    num = (num + 2.209460984245205e2) * z;
    num = (num - 2.759285104469687e2) * z;
    num = (num + 1.383577518672690e2) * z;
    num = (num - 3.066479806614716e1) * z;
    num += 2.506628277459239;
    den = (den + 1.615858368580409e2) * z;
    den = (den - 1.556989798598866e2) * z;
    den = (den + 6.680131188771972e1) * z;
    den = (den - 1.328068155288572e1) * z;
    den += 1.0;
    return num / den;
}

double qnorm_ackalm(double p, double m, double s) {
    double z = p < 0.02425 ? sqrt(-2.0 * log(p)) : \
              (p > 0.97575 ? sqrt(-2.0 * log1p(-p)) : \
              (p - 0.5) * (p - 0.5));
    s *= p < 0.02425 ? 1.0 : \
        (p > 0.97575 ? -1.0 : \
              p - 0.5);
    z = p < 0.02425 || p > 0.97575 ? R1(z) : R2(z);
    return s * z + m;
}

/* Third rational Chebyshev approximation $R_{3(7,7)}(z)$ */
static inline double R3(double z) {
    double num = 2.5090809287301226727e3 * z;
    double den = 5.2264952788528545610e3 * z;
    num = (num + 3.3430575583588128105e4) * z;
    num = (num + 6.7265770927008700853e4) * z;
    num = (num + 4.5921953931549871457e4) * z;
    num = (num + 1.3731693765509461125e4) * z;
    num = (num + 1.9715909503065514427e3) * z;
    num = (num + 1.3314166789178437745e2) * z;
    num += 3.3871328727963666080;
    den = (den + 2.8729085735721942674e4) * z;
    den = (den + 3.9307895800092710610e4) * z;
    den = (den + 2.1213794301586595867e4) * z;
    den = (den + 5.3941960214247511077e3) * z;
    den = (den + 6.8718700749205790830e2) * z;
    den = (den + 4.2313330701600911252e1) * z;
    den += 1.0;
    return num / den;
}

/* Fourth rational Chebyshev approximation $R_{4(7,7)}(z)$ */
static inline double R4(double z) {
    double num = 2.01033439929228813265e-7 * z;
    double den = 2.04426310338993978564e-15 * z;
    num = (num + 2.71155556874348757815e-5) * z;
    num = (num + 1.24266094738807843860e-3) * z;
    num = (num + 2.65321895265761230930e-2) * z;
    num = (num + 2.96560571828504891230e-1) * z;
    num = (num + 1.78482653991729133580) * z;
    num = (num + 5.46378491116411436990) * z;
    num += 6.65790464350110377720;
    den = (den + 1.42151175831644588870e-7) * z;
    den = (den + 1.84631831751005468180e-5) * z;
    den = (den + 7.86869131145613259100e-4) * z;
    den = (den + 1.48753612908506148525e-2) * z;
    den = (den + 1.36929880922735805310e-1) * z;
    den = (den + 5.99832206555887937690e-1) * z;
    den += 1.0;
    return num / den;
}

/* Fifth rational Chebyshev approximation $R_{5(7,7)}(z)$ */
static inline double R5(double z) {
    double num = 7.74545014278341407640e-4 * z;
    double den = 1.05075007164441684324e-9 * z;
    num = (num + 2.27238449892691845833e-2) * z;
    num = (num + 2.41780725177450611770e-1) * z;
    num = (num + 1.27045825245236838258) * z;
    num = (num + 3.64784832476320460504) * z;
    num = (num + 5.76949722146069140550) * z;
    num = (num + 4.63033784615654529590) * z;
    num += 1.42343711074968357734;
    den = (den + 5.47593808499534494600e-4) * z;
    den = (den + 1.51986665636164571966e-2) * z;
    den = (den + 1.48103976427480074590e-1) * z;
    den = (den + 6.89767334985100004550e-1) * z;
    den = (den + 1.67638483018380384940) * z;
    den = (den + 2.05319162663775882187) * z;
    den += 1.0;
    return num / den;
}

double qnorm_whichura(double p, double m, double s) { /** Best approximation in this file */
    int wht = (p > EM25) + (p >= 0.075) + (p > 0.925) + (p >= NEM1M25);
    double z;
    switch (wht)
    {
    case 0:
        z = R4(sqrt(-log(p)) - 5.0);
        break;
    case 1:
        z = R5(sqrt(-log(p)) - 1.6);
        break;
    case 2:
        z = 0.425 * 0.425 - (p - 0.5) * (p - 0.5);
        z = R3(z);
        break;
    case 3:
        z = R5(sqrt(-log1p(-p)) - 1.6);
        break;
    case 4:
        z = R4(sqrt(-log1p(-p)) - 5.0);
        break;
    default:
        z = nan("");
        break;
    }
    s *= p < 0.075 ? -1.0 : (p > 0.925 ? 1.0 : p - 0.5);
    return s * z + m;
}

/** NOTE: The following functions are used for low-accuracy approximation of the qnorm function */
double qnorm_odeh_n_evans(double p, double m, double s) { /* Most accurate among the low-accuracy approximations */
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = sqrt(-2.0 * log(r));
        z = y - ((((4.53642210148e-5 * y + 0.0204231210245) * y + 0.342242088547) * y + 1.0) * y + 0.322232431088) / ((((0.0038560700634 * y + 0.10353775285) * y + 0.531103462366) * y + 0.588581570495) * y + 0.099348462606);
    }
    return sgn * z * s + m;
}

double qnorm_beasley_n_springer_complete(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    double q;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        q = p - 0.5;
        if (fabs(q) <= 0.42) {
            r = q * q;
            z = q * (((25.44106 * r - 41.3912) * r + 18.615) * r - 2.506628) / ((((3.130829 * r - 21.06224) * r + 23.08337) * r - 8.473511) * r  + 1.0);
            sgn = 1.0;
        }
        else {
            r = sqrt(-log(r));
            z = (((2.321213 * r + 4.850141) * r - 2.297965) * r - 2.787189) / ((1.637068 * r + 3.543889) * r + 1.0);
        }
    }
    return sgn * z * s + m;
}

double qnorm_hastings_67(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = sqrt(-2.0 * log(r));
        z = y - (0.27061 * y + 2.30753) / ((0.04481 * y + 0.99229) * y + 1.0);
    }
    return sgn * z * s + m;
}

double qnorm_hastings_68(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = sqrt(-2.0 * log(r));
        z = y - ((0.010328 * y + 0.802853) * y + 2.515517) / (((0.001308 * y + 0.189269) * y + 1.432788) * y + 1.0);
    }
    return sgn * z * s + m;
}

double qnorm_hill_n_davis(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = sqrt(-2.0 * log(r));
        z = y - ((7.45551 * y + 450.636) * y + 1271.059) / (((y + 110.4212) * y + 750.365) * y + 500.756);
    }
    return sgn * z * s + m;
}

double qnorm_beasley_n_springer_tail(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        r = sqrt(-log(r));
        z = (((2.321213 * r + 4.850141) * r - 2.297965) * r - 2.787189) / ((1.637068 * r + 3.543889) * r + 1.0);
    }
    return sgn * z * s + m;
}

double qnorm_bailey_complete(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, u, r = (double) (p > 0.5) + p * sgn;
    double const w = 1.570796;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        if (r < 2.2e-6) {
            u = -2.0 * log(r);
            y = sqrt(u - log(4 * w * u));
            z = y + (0.5962 + 0.1633 * y) / (y * y * y);
        }
        else {
            y = -w * log(4.0 * p * (1.0 - p));
            if (y < 0) {
                z = 0.0;
            }
            else {
                z = (((4.3728e-6 * y - 2.881e-4) * y + 0.0078365) * y + 1.0) * sqrt(y);
            }
        }
    }
    return sgn * z * s + m;
}

double qnorm_bailey_central(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    double const w = 1.570796;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = -w * log(4.0 * p * (1.0 - p));
        if (y < 0) {
            z = 0.0;
        }
        else {
            z = (((4.3728e-6 * y - 2.881e-4) * y + 0.0078365) * y + 1.0) * sqrt(y);
        }
    }
    return sgn * z * s + m;
}

double qnorm_koehler(double p, double m, double s) {
    double z, sgn = (1.0 - 2.0 * (double) (p > 0.5));
    double y, r = (double) (p > 0.5) + p * sgn;
    if (r < 1e-20) {
        z = 20.0;
    }
    else {
        y = -log(4.0 * p * (1.0 - p));
        if (y < 0) {
            z = 0.0;
        }
        else {
            y = sqrt(y);
            z = y / (0.81 - 0.0193 * y);
        }
    }
    return sgn * z * s + m;
}


/* Test function */
int main() {
    double x = -1.64;
    double d, p, q;
    double tmp;
    d = dnorm(x, 0.0, 1.0);
    p = pnorm(x, 0.0, 1.0);
    q = qnorm(0.95, 0.0, 1.0);
    printf("x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    /* Main function to test the random generation of a normal variable */
    for (int i = 1; i <= 40; i++) {
        tmp = rnorm(0.0, 1.0);
        if (tmp >= 0.0) printf(" ");
        printf("%f\t", tmp);
        if (i % 5 == 0) printf("\n");
    }
    printf("High-accuracy approximations:\n");
    q = qnorm_ackalm(0.95, 0.0, 1.0);
    printf("Ackalm: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_whichura(0.95, 0.0, 1.0);
    printf("Whichura: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    printf("Low-accuracy approximations:\n");
    q = qnorm_hastings_67(0.95, 0.0, 1.0);
    printf("Hastings (67) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_hastings_68(0.95, 0.0, 1.0);
    printf("Hastings (68) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_hill_n_davis(0.95, 0.0, 1.0);
    printf("Hill and Davis approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_odeh_n_evans(0.95, 0.0, 1.0);
    printf("Odeh and Evans approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_beasley_n_springer_complete(0.95, 0.0, 1.0);
    printf("Beasley and Springer (complete) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_beasley_n_springer_tail(0.95, 0.0, 1.0);
    printf("Beasley and Springer (tail) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_bailey_complete(0.95, 0.0, 1.0);
    printf("Bailey (complete) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_bailey_central(0.95, 0.0, 1.0);
    printf("Bailey (central) approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    q = qnorm_koehler(0.95, 0.0, 1.0);
    printf("Koehler approximation: x = %f, d = %f, p = %f, q = %f\n", x, d, p, q);
    return 0;
}
