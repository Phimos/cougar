#ifndef __ROLLING_RANK_H__
#define __ROLLING_RANK_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

struct treap_node_ {
    double value_;                // value of the node
    struct treap_node_* parent_;  // parent node
    struct treap_node_* left_;    // left child
    struct treap_node_* right_;   // right child
    size_t size_;                 // size of the subtree
    size_t count_;                // number of nodes with the same value
    size_t priority_;             // priority of the node
};

struct treap_ {
    struct treap_node_* buffer_;
    struct treap_node_* buffer_end_;
    struct treap_node_* root_;
    size_t size_;
    struct treap_node_* head_;
    struct treap_node_* tail_;
};

void treap_reset(struct treap_* treap) {
    treap->root_ = NULL;
    treap->size_ = 0;
    treap->head_ = treap->tail_ = NULL;
}

struct treap_* treap_init(size_t size) {
    struct treap_* treap = malloc(sizeof(struct treap_));
    treap->buffer_ = malloc(sizeof(struct treap_node_) * size);
    treap->buffer_end_ = treap->buffer_ + size - 1;

    treap_reset(treap);
    return treap;
}

struct treap_node_* get_valid_parent(struct treap_node_* node) {
    return node->parent_ = (node->parent_->count_ == 0) ? get_valid_parent(node->parent_) : node->parent_;
}

void treap_free(struct treap_* treap) {
    free(treap->buffer_);
    free(treap);
}

void treap_print(struct treap_node_* current) {
    if (current == NULL) {
        return;
    }
    if (current->left_ != NULL && current->left_->parent_ != current) {
        printf("ERROR: left parent error %lf -> %lf\n", current->value_, current->left_->value_);
    }
    if (current->right_ != NULL && current->right_->parent_ != current) {
        printf("ERROR: right parent error %lf -> %lf\n", current->value_, current->right_->value_);
    }

    treap_print(current->left_);
    treap_print(current->right_);
}

void treap_update(struct treap_node_* current) {
    current->size_ = current->count_;
    if (current->left_ != NULL) {
        current->size_ += current->left_->size_;
    }
    if (current->right_ != NULL) {
        current->size_ += current->right_->size_;
    }
}

void treap_left_rotate(struct treap_node_** current) {
    struct treap_node_* right = (*current)->right_;
    (*current)->right_ = right->left_;
    if (right->left_ != NULL) {
        right->left_->parent_ = *current;
    }
    right->left_ = *current;
    right->parent_ = (*current)->parent_;
    (*current)->parent_ = right;

    treap_update(*current);
    treap_update(right);
    if (right->parent_ != NULL) {
        if (right->parent_->left_ == *current) {
            right->parent_->left_ = right;
        } else {
            right->parent_->right_ = right;
        }
    }
    *current = right;
}

void treap_right_rotate(struct treap_node_** current) {
    struct treap_node_* left = (*current)->left_;
    (*current)->left_ = left->right_;
    if (left->right_ != NULL) {
        left->right_->parent_ = *current;
    }
    left->right_ = *current;
    left->parent_ = (*current)->parent_;
    (*current)->parent_ = left;

    treap_update(*current);
    treap_update(left);
    if (left->parent_ != NULL) {
        if (left->parent_->left_ == *current) {
            left->parent_->left_ = left;
        } else {
            left->parent_->right_ = left;
        }
    }
    *current = left;
}

struct treap_node_* node_init(struct treap_* treap, double value) {
    if (treap->head_ == NULL) {
        treap->head_ = treap->tail_ = treap->buffer_;
    } else {
        if (treap->tail_ == treap->buffer_end_) {
            treap->tail_ = treap->buffer_;
        } else {
            ++(treap->tail_);
        }
    }

    struct treap_node_* node = treap->tail_;
    node->value_ = value;
    node->parent_ = node->left_ = node->right_ = NULL;
    node->size_ = node->count_ = 1;
    node->priority_ = rand();
    return node;
}

double query_rank(struct treap_* treap, struct treap_node_* node) {
    if (node->count_ == 0) {
        node = get_valid_parent(node);
    }
    double rank = 0;
    struct treap_node_* current = treap->root_;
    while (current != node) {
        if (node->value_ < current->value_) {
            current = current->left_;
        } else if (node->value_ > current->value_) {
            rank += current->count_ + ((current->left_ == NULL) ? 0 : current->left_->size_);
            current = current->right_;
        } else {
            // never reach
            printf("never reach\n");
        }
    }
    rank += ((current->left_ == NULL) ? 0 : current->left_->size_) + (double)(current->count_ + 1) / 2;
    return rank;
}

void treap_insert_node(struct treap_node_** current, struct treap_node_* node, struct treap_node_* parent) {
    if (*current == NULL) {
        *current = node;
        node->parent_ = parent;
        return;
    }

    if (node->value_ < (*current)->value_) {
        treap_insert_node(&((*current)->left_), node, *current);
        if ((*current)->left_->priority_ < (*current)->priority_) {
            treap_right_rotate(current);
        }
    } else if (node->value_ > (*current)->value_) {
        treap_insert_node(&((*current)->right_), node, *current);
        if ((*current)->right_->priority_ < (*current)->priority_) {
            treap_left_rotate(current);
        }
    } else {
        node->count_ = (*current)->count_ + 1;
        node->parent_ = (*current)->parent_;
        node->left_ = (*current)->left_;
        node->right_ = (*current)->right_;
        node->priority_ = (*current)->priority_;

        if ((*current)->left_ != NULL) {
            (*current)->left_->parent_ = node;
        }

        if ((*current)->right_ != NULL) {
            (*current)->right_->parent_ = node;
        }

        (*current)->count_ = 0;
        (*current)->parent_ = node;
        *current = node;
    }
    treap_update(*current);
}

double treap_insert(struct treap_* treap, double value) {
    struct treap_node_* node = node_init(treap, value);
    treap_insert_node(&(treap->root_), node, NULL);
    ++(treap->size_);
    return query_rank(treap, node);
}

struct treap_node_* update_till_root(struct treap_node_* current) {
    if (current == NULL) {
        return NULL;
    }
    while (current->parent_ != NULL) {
        treap_update(current);
        current = current->parent_;
    }
    treap_update(current);
    return current;
}

struct treap_node_* treap_delete_node(struct treap_node_* current) {
    if (current->count_ > 1) {
        --(current->count_);
        return update_till_root(current);
    }

    if (current->left_ == NULL && current->right_ == NULL) {
        if (current->parent_) {
            if (current->parent_->left_ == current) {
                current->parent_->left_ = NULL;
            } else {
                current->parent_->right_ = NULL;
            }
        }
        return update_till_root(current->parent_);
    }

    if (current->left_ == NULL) {
        if (current->parent_) {
            if (current->parent_->left_ == current) {
                current->parent_->left_ = current->right_;
            } else {
                current->parent_->right_ = current->right_;
            }
        }
        current->right_->parent_ = current->parent_;
        return update_till_root(current->right_);
    }

    if (current->right_ == NULL) {
        if (current->parent_) {
            if (current->parent_->left_ == current) {
                current->parent_->left_ = current->left_;
            } else {
                current->parent_->right_ = current->left_;
            }
        }
        current->left_->parent_ = current->parent_;
        return update_till_root(current->left_);
    }

    if (current->left_->priority_ < current->right_->priority_) {
        treap_right_rotate(&current);
        return treap_delete_node(current->right_);
    } else {
        treap_left_rotate(&current);
        return treap_delete_node(current->left_);
    }
}

void treap_delete(struct treap_* treap) {
    struct treap_node_* node = treap->head_;
    if (treap->head_ == treap->tail_) {
        treap->head_ = treap->tail_ = NULL;
    } else {
        if (treap->head_ == treap->buffer_end_) {
            treap->head_ = treap->buffer_;
        } else {
            ++(treap->head_);
        }
    }
    --(treap->size_);

    if (node->count_ == 0) {
        treap->root_ = treap_delete_node(get_valid_parent(node));
    } else {
        treap->root_ = treap_delete_node(node);
    }
}
static void rolling_rank_float64(PyArrayObject* input, PyArrayObject* output, int window, int min_count, int axis) {
    Py_ssize_t n = PyArray_SHAPE(input)[axis];
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];

    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);

    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;
    int count = 0, i = 0;
    npy_float64 curr, prev;
    npy_float64 rank;

    struct treap_* treap = treap_init(window + 1);

    Py_BEGIN_ALLOW_THREADS;
    while (input_iter->index < input_iter->size) {
        prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);
        output_ptr = PyArray_ITER_DATA(output_iter);
        count = 0;

        treap_reset(treap);

        for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            if (npy_isfinite(curr)) {
                treap_insert(treap, curr);
                ++count;
            }
            *((npy_float64*)output_ptr) = NPY_NAN;
        }

        for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            if (npy_isfinite(curr)) {
                rank = treap_insert(treap, curr);
                ++count;
            }
            *((npy_float64*)output_ptr) = count >= min_count ? (rank - 1) / (count - 1) * 2.0 - 1.0 : NPY_NAN;
        }

        for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            prev = *((npy_float64*)prev_ptr);

            if (npy_isfinite(prev)) {
                treap_delete(treap);
                --count;
            }

            if (npy_isfinite(curr)) {
                rank = treap_insert(treap, curr);
                ++count;
            }

            *((npy_float64*)output_ptr) = count >= min_count ? (rank - 1) / (count - 1) * 2.0 - 1.0 : NPY_NAN;
        }

        PyArray_ITER_NEXT(input_iter);
        PyArray_ITER_NEXT(output_iter);
    }
    treap_free(treap);
    Py_END_ALLOW_THREADS;
}

static PyObject* rolling_rank(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *median = NULL;
    if (PyArray_Check(input)) {
        arr = (PyArrayObject*)input;
        Py_INCREF(arr);
    } else {
        arr = (PyArrayObject*)PyArray_FROM_O(input);
        if (arr == NULL) {
            return NULL;
        }
    }

    int dtype = PyArray_TYPE(arr);
    int ndim = PyArray_NDIM(arr);

    min_count = min_count < 0 ? window : min_count;
    axis = axis < 0 ? ndim + axis : axis;

    if (dtype == NPY_FLOAT32)
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT32, 0);
    else
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT64, 0);
    median = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_rank_float64(arr, median, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif