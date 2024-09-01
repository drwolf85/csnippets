#ifndef UNI_TABLE_H
#define UNI_TABLE_H

#include <stdint.h>

typedef struct univariate_table {
	uint32_t *category;
	uint32_t *count;
	uint32_t tab_size;
} uni_table;

#endif

