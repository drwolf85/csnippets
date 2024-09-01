#ifndef DATA_DICT_H
#define DATA_DICT_H

#include "vartype.h"

typedef struct data_dictionary {
	char *var_name;
	vartype type;
	double scale[2];
	char *var_desc;
} data_dict;

#endif

