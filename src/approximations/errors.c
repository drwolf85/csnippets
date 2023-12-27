double absolute_error(double approx, double truth) {
    approx -= truth;
    return approx;
}

double relative_error(double approx, double truth) {
    approx -= truth;
    return approx / truth;
}

double approx_value(double truth, double rel_err) {
    truth *= 1.0 + rel_err;
    return truth;
}
