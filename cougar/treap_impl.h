#ifdef T

#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define concat(a, b) a##_##b

#define treap_node(dtype) struct concat(treap_node, dtype)
#define treap(dtype) struct concat(treap, dtype)
#define method(name, dtype) concat(treap_##name, dtype)

treap_node(T) {
    T value;
    treap_node(T) * parent;
    treap_node(T) * children[2];
    size_t size, count, priority;
};

treap(T) {
    treap_node(T) * buffer;
    treap_node(T) * buffer_end;

    treap_node(T) * root;
    size_t size, capacity;

    treap_node(T) * head, *tail;
};

static inline void method(reset, T)(treap(T) * treap) {
    treap->root = NULL;
    treap->size = 0;
    treap->head = treap->tail = NULL;
}

static treap(T) * method(init, T)(size_t capacity) {
    treap(T)* treap = (treap(T)*)malloc(sizeof(treap(T)));
    treap->buffer = (treap_node(T)*)malloc(sizeof(treap_node(T)) * capacity);
    treap->buffer_end = treap->buffer + capacity - 1;
    treap->capacity = capacity;
    method(reset, T)(treap);
    return treap;
}

static inline void method(free, T)(treap(T) * treap) {
    free(treap->buffer);
    free(treap);
}

static inline void method(update, T)(treap_node(T) * node) {
    node->size = node->count;
    if (node->children[0]) {
        node->size += node->children[0]->size;
    }
    if (node->children[1]) {
        node->size += node->children[1]->size;
    }
}

static inline int method(is_left_child, T)(treap_node(T) * node) {
    return node->parent->children[0] == node;
}

static inline int method(is_right_child, T)(treap_node(T) * node) {
    return node->parent->children[1] == node;
}

static inline int method(is_root, T)(treap_node(T) * node) {
    return node->parent == NULL;
}

static inline treap_node(T) * method(rotate, T)(treap_node(T) * node, int direction) {
    // direction = 0: left rotate
    // direction = 1: right rotate

    treap_node(T)* parent = node->parent;
    treap_node(T)* child = node->children[direction ^ 1];
    treap_node(T)* grandchild = child->children[direction];

    child->parent = parent;
    if (parent) {
        parent->children[method(is_right_child, T)(node)] = child;
    }

    node->parent = child;
    child->children[direction] = node;

    node->children[direction ^ 1] = grandchild;
    if (grandchild) {
        grandchild->parent = node;
    }

    method(update, T)(node);
    method(update, T)(child);
    return child;
}

static inline treap_node(T) * method(left_rotate, T)(treap_node(T) * node) {
    return method(rotate, T)(node, 0);
}

static inline treap_node(T) * method(right_rotate, T)(treap_node(T) * node) {
    return method(rotate, T)(node, 1);
}

static inline treap_node(T) * method(step, T)(treap(T) * treap, treap_node(T) * node) {
    return (node == treap->buffer_end) ? treap->buffer : node + 1;
}

static inline treap_node(T) * method(init_node, T)(treap(T) * treap, T value) {
    if (treap->head == NULL) {
        treap->head = treap->tail = treap->buffer;
    } else {
        treap->tail = method(step, T)(treap, treap->tail);
    }

    treap_node(T)* node = treap->tail;
    node->value = value;
    node->parent = node->children[0] = node->children[1] = NULL;
    node->size = node->count = 1;
    node->priority = rand();
    return node;
}

static inline treap_node(T) * method(insert_node, T)(treap_node(T) * current, treap_node(T) * node) {
    if (current == NULL) {
        return node;
    }

    if (node->value < current->value) {
        current->children[0] = method(insert_node, T)(current->children[0], node);
        if (current->children[0]->priority < current->priority) {
            current = method(right_rotate, T)(current);
        }
    } else if (node->value > current->value) {
        current->children[1] = method(insert_node, T)(current->children[1], node);
        if (current->children[1]->priority < current->priority) {
            current = method(left_rotate, T)(current);
        }
    } else {
        node->count = current->count + 1;
        node->children[0] = current->children[0];
        node->children[1] = current->children[1];
        node->parent = current->parent;

        if (node->children[0])
            node->children[0]->parent = node;
        if (node->children[1])
            node->children[1]->parent = node;

        current->count = 0;
        current->parent = node;
    }

    method(update, T)(current);
    return current;
}

static inline void method(insert, T)(treap(T) * treap, T value) {
    treap_node(T)* node = method(init_node, T)(treap, value);
    treap->root = method(insert_node, T)(treap->root, node);
    treap->size++;
}

static inline double method(query_rank, T)(treap(T) * treap) {
    double rank = 0;
    treap_node(T)* current = treap->root;
    treap_node(T)* node = treap->tail;

    while (1) {
        if (node->value < current->value) {
            current = current->children[0];
        } else if (node->value > current->value) {
            rank += (double)(current->count);
            if (current->children[0]) {
                rank += (double)(current->children[0]->size);
            }
            current = current->children[1];
        } else {
            rank += ((double)((current->count) + 1)) / 2;
            if (current->children[0]) {
                rank += (double)(current->children[0]->size);
            }
            return rank;
        }
    }
}

static inline T method(query_kth, T)(treap(T) * treap, size_t k) {
    assert((k > 0) && (k <= treap->size));
    treap_node(T)* current = treap->root;
    size_t left_subtree_size = 0;
    while (1) {
        left_subtree_size = current->children[0] ? current->children[0]->size : 0;
        if (k <= left_subtree_size) {
            current = current->children[0];
        } else if (k > left_subtree_size + current->count) {
            k -= (left_subtree_size + current->count);
            current = current->children[1];
        } else {
            return current->value;
        }
    }
}

static inline T method(query_quantile, T)(treap(T) * treap, double quantile) {
    assert((quantile >= 0) && (quantile <= 1));
    double rank = quantile * (double)(treap->size - 1) + 1;
    size_t lower = (size_t)floor(rank);
    size_t upper = (size_t)ceil(rank);

    if (lower == upper) {
        return method(query_kth, T)(treap, lower);
    } else {
        double lower_weight = (double)(upper)-rank;
        double upper_weight = rank - (double)(lower);
        return lower_weight * method(query_kth, T)(treap, lower) + upper_weight * method(query_kth, T)(treap, upper);
    }
}

static inline treap_node(T) * method(update_to_root, T)(treap_node(T) * node) {
    if (node == NULL) {
        return NULL;
    }
    while (node->parent) {
        method(update, T)(node);
        node = node->parent;
    }
    method(update, T)(node);
    return node;
}

static inline treap_node(T) * method(remove_node, T)(treap_node(T) * node) {
    if (node->count > 1) {
        --(node->count);
        return method(update_to_root, T)(node);
    }

    if ((node->children[0] == NULL) && (node->children[1] == NULL)) {
        if (node->parent) {
            node->parent->children[method(is_right_child, T)(node)] = NULL;
        }
        return method(update_to_root, T)(node->parent);
    }

    if (node->children[0] == NULL) {
        node->children[1]->parent = node->parent;
        if (node->parent) {
            node->parent->children[method(is_right_child, T)(node)] = node->children[1];
        }
        return method(update_to_root, T)(node->children[1]);
    }

    if (node->children[1] == NULL) {
        node->children[0]->parent = node->parent;
        if (node->parent) {
            node->parent->children[method(is_right_child, T)(node)] = node->children[0];
        }
        return method(update_to_root, T)(node->children[0]);
    }

    if (node->children[0]->priority < node->children[1]->priority) {
        node = method(right_rotate, T)(node);
        return method(remove_node, T)(node->children[1]);
    } else {
        node = method(left_rotate, T)(node);
        return method(remove_node, T)(node->children[0]);
    }
}

static inline treap_node(T) * method(find_node, T)(treap_node(T) * node) {
    return node->parent = (node->parent->count) ? node->parent : method(find_node, T)(node->parent);
}

static inline void method(remove, T)(treap(T) * treap) {
    treap_node(T)* node = treap->head;
    --(treap->size);
    if (treap->head == treap->tail) {
        treap->head = treap->tail = NULL;
    } else {
        treap->head = method(step, T)(treap, treap->head);
    }

    // if (node->count == 0) {
    //     treap->root = method(remove_node, T)(method(find_node, T)(node));
    // } else {
    //     treap->root = method(remove_node, T)(node);
    // }

    if (node->count == 0) {
        node = method(find_node, T)(node);
    }
    treap->root = method(remove_node, T)(node);
}

#undef method
#undef treap_node
#undef treap
#undef concat

#endif