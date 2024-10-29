#ifndef HYPERGRAPH_H
#define HYPERGRAPH_H

#include <stdint.h>

typedef struct hyper_vertex_def {
	uint64_t id; /* To store the vertext ID */
	void *attr; /* To allow for storage of potential attirbutes */
} hyper_vertex_t;

typedef struct hyper_edge_def {
	uint64_t *nodes; /* Position matters (i.e., {0: parent; 1: child} */
	uint64_t node_size;
	void *attr; /* To allow for storage of potential attirbutes */
} hyper_edge_t;

typedef struct hyper_graph_def {
	hyper_vertex_t *node;
	uint64_t node_list_size;
	hyper_edge_t *edge;
	uint64_t edge_list_size;
} hyper_graph_t;

#endif
