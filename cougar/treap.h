#ifndef __TREAP_H__
#define __TREAP_H__

#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "debug.h"

struct treap_node_ {
    double value_;                     // value of the node
    struct treap_node_* parent_;       // parent node
    struct treap_node_* children_[2];  // left and right children
    size_t size_;                      // size of the subtree
    size_t count_;                     // number of nodes with the same value
    size_t priority_;                  // priority of the node
};

struct treap_ {
    struct treap_node_* buffer_;      // buffer for nodes
    struct treap_node_* buffer_end_;  // end of the buffer

    struct treap_node_* root_;
    size_t size_, capacity_;
    struct treap_node_ *head_, *tail_;
};

static inline void treap_reset(struct treap_* treap) {
    treap->root_ = NULL;
    treap->size_ = 0;
    treap->head_ = treap->tail_ = NULL;
}

static struct treap_* treap_init(size_t capacity) {
    struct treap_* treap = (struct treap_*)malloc(sizeof(struct treap_));
    treap->buffer_ = (struct treap_node_*)malloc(sizeof(struct treap_node_) * (capacity));
    treap->buffer_end_ = treap->buffer_ + capacity - 1;

    treap_reset(treap);
    return treap;
}

static void treap_free(struct treap_* treap) {
    free(treap->buffer_);
    free(treap);
}

static inline void treap_update(struct treap_node_* node) {
    node->size_ = node->count_;
    if (node->children_[0])
        node->size_ += node->children_[0]->size_;
    if (node->children_[1])
        node->size_ += node->children_[1]->size_;
}

static inline int treap_is_left_child(struct treap_node_* node) {
    return node->parent_->children_[0] == node;
}

static inline int treap_is_right_child(struct treap_node_* node) {
    return node->parent_->children_[1] == node;
}

static inline int treap_is_root(struct treap_node_* node) {
    return node->parent_ == NULL;
}

static inline struct treap_node_* treap_rotate(struct treap_node_* node, int direction) {
    // direction = 0: left rotate
    // direction = 1: right rotate

    struct treap_node_* parent = node->parent_;
    struct treap_node_* child = node->children_[direction ^ 1];
    struct treap_node_* grandchild = child->children_[direction];

    child->parent_ = parent;
    if (parent)
        parent->children_[treap_is_right_child(node)] = child;

    node->parent_ = child;
    child->children_[direction] = node;

    node->children_[direction ^ 1] = grandchild;
    if (grandchild)
        grandchild->parent_ = node;

    treap_update(node);
    treap_update(child);
    return child;
}

static inline struct treap_node_* treap_left_rotate(struct treap_node_* node) {
    return treap_rotate(node, 0);
}

static inline struct treap_node_* treap_right_rotate(struct treap_node_* node) {
    return treap_rotate(node, 1);
}

static inline struct treap_node_* treap_step(struct treap_* treap, struct treap_node_* node) {
    return node == treap->buffer_end_ ? treap->buffer_ : node + 1;
}

static inline struct treap_node_* treap_node_init(struct treap_* treap, double value) {
    if (treap->head_ == NULL) {
        treap->head_ = treap->tail_ = treap->buffer_;
    } else {
        treap->tail_ = treap_step(treap, treap->tail_);
    }

    struct treap_node_* node = treap->tail_;
    node->value_ = value;
    node->parent_ = node->children_[0] = node->children_[1] = NULL;
    node->size_ = node->count_ = 1;
    node->priority_ = rand();
    return node;
}

static inline struct treap_node_* treap_node_insert(struct treap_node_* current, struct treap_node_* node, struct treap_node_* parent) {
    if (current == NULL) {
        return node;
    }

    if (node->value_ < current->value_) {
        current->children_[0] = treap_node_insert(current->children_[0], node, current);
        if (current->children_[0]->priority_ < current->priority_) {
            current = treap_right_rotate(current);
        }
    } else if (node->value_ > current->value_) {
        current->children_[1] = treap_node_insert(current->children_[1], node, current);
        if (current->children_[1]->priority_ < current->priority_) {
            current = treap_left_rotate(current);
        }
    } else {
        node->count_ = current->count_ + 1;
        node->children_[0] = current->children_[0];
        node->children_[1] = current->children_[1];
        node->parent_ = current->parent_;

        if (node->children_[0])
            node->children_[0]->parent_ = node;
        if (node->children_[1])
            node->children_[1]->parent_ = node;

        current->count_ = 0;
        current->parent_ = node;
    }
    treap_update(current);
    return current;
}

static inline void treap_insert(struct treap_* treap, double value) {
    debug("[treap] insert %lf\n", value);
    struct treap_node_* node = treap_node_init(treap, value);
    treap->root_ = treap_node_insert(treap->root_, node, NULL);
    treap->size_++;
}

static inline double treap_query_rank(struct treap_* treap) {
    double rank = 0;
    struct treap_node_* current = treap->root_;
    struct treap_node_* node = treap->tail_;

    while (1) {
        if (node->value_ < current->value_) {
            current = current->children_[0];
        } else if (node->value_ > current->value_) {
            rank += current->count_;
            if (current->children_[0])
                rank += current->children_[0]->size_;
            current = current->children_[1];
        } else {
            rank += (double)(current->count_ + 1) / 2;
            if (current->children_[0])
                rank += current->children_[0]->size_;
            return rank;
        }
    }
}

static inline double treap_query_kth(struct treap_* treap, size_t k) {
    debug("[treap] query kth %zu\n", k);
    assert((k > 0) && (k <= treap->size_));
    struct treap_node_* current = treap->root_;
    size_t left_subtree_size = 0;
    while (1) {
        left_subtree_size = (current->children_[0]) ? current->children_[0]->size_ : 0;
        if (k <= left_subtree_size) {
            current = current->children_[0];
        } else if (k > left_subtree_size + current->count_) {
            k -= left_subtree_size + current->count_;
            current = current->children_[1];
        } else {
            return current->value_;
        }
    }
}

static inline double treap_query_quantile(struct treap_* treap, double quantile) {
    debug("[treap] query quantile %lf\n", quantile);
    assert((quantile >= 0) && (quantile <= 1));
    double rank = quantile * (treap->size_ - 1) + 1;
    size_t lower = floor(rank);
    size_t upper = ceil(rank);

    if (lower == upper) {
        return treap_query_kth(treap, lower);
    } else {
        return (treap_query_kth(treap, lower) * (upper - rank) + treap_query_kth(treap, upper) * (rank - lower));
    }
}

static inline struct treap_node_* treap_update_to_root(struct treap_node_* node) {
    if (node == NULL)
        return NULL;
    while (node->parent_) {
        treap_update(node);
        node = node->parent_;
    }
    treap_update(node);
    return node;
}

static inline struct treap_node_* treap_node_remove(struct treap_node_* node) {
    debug("[treap] node remove %lf\n", node->value_);
    if (node->count_ > 1) {
        --(node->count_);
        return treap_update_to_root(node);
    }

    if (node->children_[0] == NULL && node->children_[1] == NULL) {
        if (node->parent_)
            node->parent_->children_[treap_is_right_child(node)] = NULL;
        return treap_update_to_root(node->parent_);
    }

    if (node->children_[0] == NULL) {
        node->children_[1]->parent_ = node->parent_;
        if (node->parent_)
            node->parent_->children_[treap_is_right_child(node)] = node->children_[1];
        return treap_update_to_root(node->children_[1]);
    }

    if (node->children_[1] == NULL) {
        node->children_[0]->parent_ = node->parent_;
        if (node->parent_)
            node->parent_->children_[treap_is_right_child(node)] = node->children_[0];
        return treap_update_to_root(node->children_[0]);
    }

    if (node->children_[0]->priority_ < node->children_[1]->priority_) {
        node = treap_right_rotate(node);
        return treap_node_remove(node->children_[1]);
    } else {
        node = treap_left_rotate(node);
        return treap_node_remove(node->children_[0]);
    }
}

static inline struct treap_node_* treap_node_find(struct treap_node_* node) {
    return node->parent_ = (node->parent_->count_) ? node->parent_ : treap_node_find(node->parent_);
}

static inline void treap_remove(struct treap_* treap) {
    struct treap_node_* node = treap->head_;
    debug("[treap] remove %lf, count = %zu\n", node->value_, node->count_);

    --(treap->size_);
    if (treap->head_ == treap->tail_) {
        treap->head_ = treap->tail_ = NULL;
    } else {
        treap->head_ = treap_step(treap, treap->head_);
    }

    if (node->count_ == 0) {
        treap->root_ = treap_node_remove(treap_node_find(node));
    } else {
        treap->root_ = treap_node_remove(node);
    }
    debug("[treap] remove done\n");
}

#endif