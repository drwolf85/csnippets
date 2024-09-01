#ifndef MULTI_TAB_H
#define MULTI_TAB_H

#include <stdint.h>

typedef struct multivariate_table {
	uint32_t **categories;
	uint32_t *counts;
	uint32_t tab_size;
	uint32_t n_vars;
} multi_tab;

#endif

