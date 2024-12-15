#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>

typedef struct basic_vertex_def {
	uint64_t id; /* To store the vertext ID */
	void *attr; /* To allow for storage of potential attirbutes */
} basic_vertex_t;

typedef struct basic_edge_def {
	uint64_t nodes[2]; /* Position matters (i.e., {0: parent; 1: child} */
	void *attr; /* To allow for storage of potential attirbutes */
} basic_edge_t;

typedef struct basic_graph_def {
	basic_vertex_t *node;
	uint64_t node_list_size;
	basic_edge_t *edge;
	uint64_t edge_list_size;
} basic_graph_t;

#endif
