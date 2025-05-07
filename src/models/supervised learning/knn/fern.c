#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DST_EPSILON 1e-12

/* Based on the algorithm proposed by Richard Zhu at Princeton University in May 7, 2024 */

typedef struct rbTree {
	struct rbTree *left;
	struct rbTree *right;
	double *data;
} rbTree;

typedef struct listNode {
	struct listNode *next;
	rbTree *tree_node;
} listNode;

typedef struct Queue {
	struct listNode *first;
	struct listNode *last;
	size_t len;
} Queue;

/**
 * FERN Insertion algorithm (based on red-black trees)
 *
 * @param node Pointer to the node of an rbTree structure
 * @param v Pointer to a vector of data
 * @param p Pointer to the length of the vector `v`
 * @param dst Pointer to a distance/dissimilarity function
 * 
 * @return the depth to reach the data point location
 */
size_t rbTree_insert(rbTree **node, double *v, size_t const *p, 
		double (*dst)(double *, double *, size_t const *)) {
	size_t dep = 0;
	if (*node == NULL) {
		*node = (rbTree *) calloc(1, sizeof(rbTree));
		(**node).data = v;
#ifdef DEBUG
		printf("\n");
#endif
		return dep;
	}
	if ((**node).left == NULL) {
#ifdef DEBUG
		printf("left ");
#endif
		dep += 1ULL + rbTree_insert(&(**node).left, v, p, NULL);
		return dep;
	}
	if ((**node).right == NULL) {
#ifdef DEBUG
		printf("right ");
#endif
		dep += 1ULL + rbTree_insert(&(**node).right, v, p, NULL);
		return dep;
	}
	if (dst) {
		if (dst((**node).left->data, v, p) < \
				dst((**node).right->data, v, p)) {
#ifdef DEBUG
			printf("left ");
#endif
			dep += 1ULL + rbTree_insert(&(**node).left, v, p, dst);
		}
		else {
#ifdef DEBUG
			printf("right ");
#endif
			dep += 1ULL + rbTree_insert(&(**node).right, v, p, dst);
		}
	}
	return dep;
}

/**
 * Free allocated memory to store the red-black tree for the FERN algorithm
 *
 * @oaram node Pointer to the node of an red-black ree structure
 */
void rbTree_free(rbTree *node) {
	if (node != NULL) {
		if (node->left) rbTree_free(node->left);
		if (node->right) rbTree_free(node->right);
		node->data = NULL;
		free(node);
	}
}

/**
 * Allocation of a list node
 *
 * @param node Pointer to a node in a red-black tree
 *
 * @return Pointer to the memory of the allocated item in the list
 */
listNode * lst_alloc(rbTree *node) {
	listNode *res = (listNode *) calloc(1, sizeof(listNode));
	res->tree_node = node;
	return res;
}

/**
 * Free items in a list
 *
 * @param lst Pointer to a list structure
 */
void lst_free(listNode *lst) {
	if (lst) {
		lst->tree_node = NULL;
		if (lst->next) lst_free(lst->next);
		free(lst);
	}
}

/**
 * Initialization of a queue
 *
 * @param queue Pointer to a queue structure
 * @param node Pointer to a node in a red-black tree
 */
void queue_init(Queue *queue, rbTree *node) {
	if (queue) {
		queue->first = lst_alloc(node);
		queue->last = queue->first;
		queue->len = 1;
	}
}

/**
 * Appending an item to a queue
 *
 * @param queue Pointer to a queue structure
 * @param node Pointer to a node in a red-black tree
 */
void queue_append(Queue *queue, rbTree *node) {
	if (queue) {
		if (queue->len == 0) {
			queue_init(queue, node);
		}
		else {
			queue->last->next = lst_alloc(node);
			queue->last = queue->last->next;
			queue->len += 1;
		}
	}
}

/**
 * Pop the first item of a queue
 *
 * @param queue Pointer to a queue structure
 * 
 * @return A pointer to the first node of a red-black tree stored in the queue
 */
rbTree * queue_pop(Queue *queue) {
	listNode *tmp;
	rbTree *ans = NULL;
	if (queue) {
		if (queue->first) ans = queue->first->tree_node;
		if (queue->len > 1) {
			tmp = queue->first->next;
			if (queue->first) {
				queue->first->tree_node = NULL;
				free(queue->first);
				queue->first = tmp;
			}
			queue->len -= 1;
		}
		if (queue->len == 0) {
			lst_free(queue->first);
			if (queue->first) free(queue->first);
			if (queue->last) free(queue->last);
		}
	}
	return ans;
}

/**
 * Find closest vector by comparison
 *
 * @param mip Pointer to a real value
 * @param mip_vec Double pointer to a vector of values of length `p`
 * @param obs_vec Pointer to a vector of observed values
 * @param query Pointer to a query vector
 * @param p Pointer to the length of there vectors mentioned above
 * @param dst Pointer to a distance/dissimilarity matrix
 */
static void closest(double *mip, double **mip_vec, double *obs_vec, 
		double *query, size_t const *p, 
		double (*dst)(double *, double *, size_t const *)) {
	double dist = (double) INFINITY;
	if (mip && mip_vec && obs_vec && query && p && dst) {
		dist = dst(obs_vec, query, p);
		if (dist < *mip) {
			*mip_vec = obs_vec;
			*mip = dist;
		}
	}
}

/**
 * Comparison function to assess if a branch can be truncated
 *
 * @param lr Pointer to a real value
 * @param query Pointer to a query vector
 * @param v1 Pointer to a data vector from a left branch
 * @param v2 Pointer to a data vector from a right branch
 * @param p Pointer to the length of the three vectors mentioned above
 * @param dst Pointer to a distance/dissimilarity function
 */
static void cmp(double *lr, double *query, 
		double *v1, double *v2, size_t const *p,
		double (*dst)(double *, double *, size_t const *)) {
	double dist = (double) INFINITY;
	double d1, d2;
	if (lr && query && v1 && v2 && p && dst) {
		dist = dst(v1, v2, p);
		if (dist <= DST_EPSILON) {
			*lr = 0.0;
			return;
		}
		d1 = dst(v1, query, p) / dist;
		d2 = dst(v2, query, p) / dist;
		*lr = d1 / (d1 + d2) - 0.5;
	}
}

/**
 * Lookup function to retrieve the nearest neighbor from the red-black tree
 *
 * @param mip_vec Pointer to a pointer to a vector of data
 * @param q Pointer to a query vector
 * @param p Pointer to the length of the query vector (and data stored in the tree)
 * @param node Pointer to the root node of the tree
 * @param max_depth Maximum length reached by a branch in the red-black tree `node`
 * @param dst Pointer to a distance/dissimilarity function
 *
 * @return the distance between the query vector and its nearest neighbor
 *         retrieved from the red-black tree. The value in `*mip_vec` will
 *         be update with the memory address of the nearest neighbor point
 */
void lookup(double **mip_vec, double *q, size_t const *p,
		rbTree *node, /* size_t max_depth, */
		double (*dst)(double *, double *, size_t const *)) {
	double mip = (double) INFINITY;
	Queue que;
	rbTree *curr;
	/* bool prune; */
	double lr = 0.0;
	if (mip_vec) {
		queue_init(&que, node);
		while (que.len > 0) {
			curr = queue_pop(&que);	/* Find closest point */
			closest(&mip, mip_vec, curr->data, q, p, dst);
			if (curr->left && curr->right) { /* Determine cmp and pruning */
				cmp(&lr, q, curr->left->data, curr->right->data, p, dst);
				if (lr <= 0.0) {
					queue_append(&que, curr->left);
					/* if (!prune) {
						queue_append(&que, curr->right);
					} */
				}
				else {
					queue_append(&que, curr->right);
					/* if (!prune) {
						queue_append(&que, curr->left);
					} */
				}
			}
			else {
				if (curr->left) {
					queue_append(&que, curr->left);
				}
				else if (curr->right) {
					queue_append(&que, curr->right);
				}
				else {
					que.len = 0;
					if (que.first) lst_free(que.first);
					return;
				}
			}
		}
	}
}

#ifdef DEBUG
double dst_euclid(double *u, double *v, size_t const *p) {
	double tmp, res = 0.0;
	size_t i;
	for (i = 0; i < *p; i++) {
		tmp = u[i] - v[i];
		res += tmp * tmp;
	}
	return sqrt(fabs(res));
}

int main(void) {
	double dta[] = { 0.4633281, 0.006397104, -1.354303, 0.02665879, \
		0.5485055, -0.4017591, -0.920428, -0.5934684, 1.069774, \
		0.2745039, -1.37654, 0.3729608, -1.899978, -1.382227, \
		-1.186784, 0.4471284, -0.1089395, -0.04377649, -0.6750611, \
		-0.2231071 };
	size_t const p = 2;
	size_t const n = 10;
	size_t i, j, k, mxd = 0;
	size_t ord[n];
	double q[2] = { 0.02665879, -1.382227 };
	double *ptr = NULL;
	rbTree *root = NULL;
	/* Shuffling step: important for high-performance during lookup */
	for (i = 0; i < n; i++) ord[i] = i;
	srand(time(NULL));
	for (i = 0; i < n; i++) {
		j = ((size_t) rand()) % n;
		k = ord[j];
		ord[j] = ord[i];
		ord[i] = k;
	}
	/* Inserting the data points in a red-black tree */
	for (i = 0; i < n; i++) {
		k = rbTree_insert(&root, &dta[ord[i] * p], &p, dst_euclid);
		if (k > mxd) mxd = k;
	}
	lookup(&ptr, q, &p, root, dst_euclid);
	rbTree_free(root);
	
	printf("Max depth: %lu\n", mxd);
	printf("    [query] %f %f\n", q[0], q[1]);
	printf("[retrieved] %f %f\n", ptr[0], ptr[1]);
	
	return 0;
}
#endif
