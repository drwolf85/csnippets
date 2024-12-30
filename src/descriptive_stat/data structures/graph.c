#include <stdint.h>
#include <stdlib.h>

#include "graph.h"

basic_vertex_t * alloc_basic_vertices(uint64_t n) {
	return (basic_vertex_t *) calloc(n, sizeof(basic_vertex_t));
}

basic_edge_t * alloc_basic_edges(uint64_t n) {
	return (basic_edge_t *) calloc(n, sizeof(basic_edge_t));
}

basic_graph_t init_graph(uint64_t n_vertices, uint64_t n_edges) {
	basic_graph_t res;
	res.node = alloc_basic_vertices(n_vertices);
	if (res.node) res.node_list_size = n_vertices;
	res.edge = alloc_basic_edges(n_edges);
	if (res.edge) res.edge_list_size = n_edges;
	return res;
}

void free_graph(basic_graph_t *g) {
	uint64_t i;
	if (g->node) {
		for (i = 0; i < g->node_list_size; i++) free(g->node[i].attr);
		free(g->node);
	}
	g->node_list_size = 0;
	if (g->edge) {
		for (i = 0; i < g->edge_list_size; i++) free(g->edge[i].attr);
		free(g->edge);
	}
	g->edge_list_size = 0;
}
