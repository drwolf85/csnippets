#ifndef DATA_DICT_H
#define DATA_DICT_H

enum vartype {nominal, ordinal, integer, real, complex, other};

typedef struct data_dictionary {
	char *var_name;
	vartype type;
	double scale[2];
	char *var_desc;
} data_dict;

#endif
