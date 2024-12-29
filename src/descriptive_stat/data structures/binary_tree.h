#ifndef BINARY_TREE_H
#define BINARY_TREE_H

typedef struct bin_tree {
	struct bin_tree *lf;
	struct bin_tree *rt;
	void *attr;
} bin_tree_t;
#endif
