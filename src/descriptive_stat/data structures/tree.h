#ifndef TREE_H
#define TREE_H

#include <stdlib.h>

typedef struct tree {
	size_t n_branches;
	struct tree *branches;
	void *attr;
} tree_t;
#endif

