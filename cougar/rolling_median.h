#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

struct node_ {
    char is_min_;   // is this node in the min heap?
    double value_;  // value of the node
    size_t index_;  // index in the heap buffer
};

struct dual_heap_ {
    struct node_* buffer_;
    struct node_* buffer_end_;
    struct node_** heap_;
    struct node_** min_heap_;
    struct node_** max_heap_;
    size_t min_size_, max_size_;
    size_t min_capacity_, max_capacity_;
    struct node_* head_;
    struct node_* tail_;
};

static inline struct dual_heap_* dual_heap_init(size_t min_heap_size, size_t max_heap_size) {
    size_t size = min_heap_size + max_heap_size;
    struct dual_heap_* heap = malloc(sizeof(struct dual_heap_));
    heap->buffer_ = malloc(sizeof(struct node_) * size);
    heap->heap_ = malloc(sizeof(struct node_*) * size);

    heap->buffer_end_ = heap->buffer_ + size - 1;

    heap->min_heap_ = heap->heap_;
    heap->max_heap_ = heap->heap_ + min_heap_size;

    heap->min_size_ = heap->max_size_ = 0;
    heap->min_capacity_ = min_heap_size;
    heap->max_capacity_ = max_heap_size;

    heap->head_ = heap->tail_ = NULL;

    return heap;
}

static inline void dual_heap_free(struct dual_heap_* heap) {
    free(heap->buffer_);
    free(heap->heap_);
    free(heap);
}

static inline void dual_heap_clear(struct dual_heap_* heap) {
    heap->min_size_ = heap->max_size_ = 0;
    heap->head_ = heap->tail_ = NULL;
}

static inline size_t get_parent(size_t index) {
    return (index - 1) >> 1;
}

static inline size_t get_left_child(size_t index) {
    return (index << 1) + 1;
}

static inline size_t get_right_child(size_t index) {
    return (index << 1) + 2;
}

static inline size_t get_smallest_child(struct node_** heap, size_t index, size_t size) {
    size_t left = get_left_child(index);
    size_t right = get_right_child(index);
    if (right < size) {
        return heap[left]->value_ < heap[right]->value_ ? left : right;
    } else if (left < size) {
        return left;
    } else {
        return size;
    }
}

static inline size_t get_largest_child(struct node_** heap, size_t index, size_t size) {
    size_t left = get_left_child(index);
    size_t right = get_right_child(index);
    if (right < size) {
        return heap[left]->value_ > heap[right]->value_ ? left : right;
    } else if (left < size) {
        return left;
    } else {
        return size;
    }
}

static inline void swap(struct node_** heap, size_t i, size_t j) {
    struct node_* temp = heap[i];
    heap[i] = heap[j];
    heap[j] = temp;

    heap[i]->index_ = i;
    heap[j]->index_ = j;
}

static inline void max_heap_sift_up(struct node_** heap, size_t index) {
    struct node_* temp = heap[index];
    size_t parent = get_parent(index);
    while (index > 0 && temp->value_ > heap[parent]->value_) {
        swap(heap, index, parent);
        index = parent;
        parent = get_parent(index);
    }
    heap[index] = temp;
}

static inline void max_heap_sift_down(struct node_** heap, size_t index, size_t size) {
    size_t child = get_largest_child(heap, index, size);
    struct node_* temp = heap[index];
    while (child < size && temp->value_ < heap[child]->value_) {
        swap(heap, index, child);
        index = child;
        child = get_largest_child(heap, index, size);
    }
    heap[index] = temp;
}

static inline void max_heap_insert(struct dual_heap_* heap, struct node_* node) {
    node->is_min_ = 0;
    node->index_ = heap->max_size_;
    heap->max_heap_[heap->max_size_] = node;
    max_heap_sift_up(heap->max_heap_, heap->max_size_);
    heap->max_size_++;
}

static inline void max_heap_delete(struct dual_heap_* heap, struct node_* node) {
    size_t index = node->index_;
    heap->max_size_--;
    swap(heap->max_heap_, index, heap->max_size_);
    max_heap_sift_up(heap->max_heap_, index);
    max_heap_sift_down(heap->max_heap_, index, heap->max_size_);
}

static inline void min_heap_sift_up(struct node_** heap, size_t index) {
    struct node_* temp = heap[index];
    size_t parent = get_parent(index);
    while (index > 0 && temp->value_ < heap[parent]->value_) {
        swap(heap, index, parent);
        index = parent;
        parent = get_parent(index);
    }
    heap[index] = temp;
}

static inline void min_heap_sift_down(struct node_** heap, size_t index, size_t size) {
    size_t child = get_smallest_child(heap, index, size);
    struct node_* temp = heap[index];
    while (child < size && temp->value_ > heap[child]->value_) {
        swap(heap, index, child);
        index = child;
        child = get_smallest_child(heap, index, size);
    }
    heap[index] = temp;
}

static inline void min_heap_insert(struct dual_heap_* heap, struct node_* node) {
    node->is_min_ = 1;
    node->index_ = heap->min_size_;
    heap->min_heap_[heap->min_size_] = node;
    min_heap_sift_up(heap->min_heap_, heap->min_size_);
    heap->min_size_++;
}

static inline void min_heap_delete(struct dual_heap_* heap, struct node_* node) {
    size_t index = node->index_;
    heap->min_size_--;
    swap(heap->min_heap_, index, heap->min_size_);
    min_heap_sift_up(heap->min_heap_, index);
    min_heap_sift_down(heap->min_heap_, index, heap->min_size_);
}

static inline void dual_heap_push(struct dual_heap_* heap, double value) {
    if (heap->head_ == NULL) {
        heap->head_ = heap->buffer_;
        heap->tail_ = heap->buffer_;
    } else {
        if (heap->tail_ == heap->buffer_end_) {
            heap->tail_ = heap->buffer_;
        } else {
            ++(heap->tail_);
        }
    }

    heap->tail_->value_ = value;

    if (heap->min_size_ <= heap->max_size_) {
        min_heap_insert(heap, heap->tail_);
    } else {
        max_heap_insert(heap, heap->tail_);
    }

    if (heap->min_size_ == 0 || heap->max_size_ == 0) {
        return;
    }

    struct node_* min_top = heap->min_heap_[0];
    struct node_* max_top = heap->max_heap_[0];

    if (min_top->value_ < max_top->value_) {
        heap->min_heap_[0] = max_top;
        heap->max_heap_[0] = min_top;

        max_top->is_min_ = 1;
        min_top->is_min_ = 0;

        min_heap_sift_down(heap->min_heap_, 0, heap->min_size_);
        max_heap_sift_down(heap->max_heap_, 0, heap->max_size_);
    }
}

static inline void dual_heap_pop(struct dual_heap_* heap) {
    if (heap->min_size_ == 0 && heap->max_size_ == 0) {
        return;
    }
    struct node_* node = heap->head_;

    if (heap->head_ == heap->tail_) {
        heap->head_ = heap->tail_ = NULL;
    } else {
        if (heap->head_ == heap->buffer_end_) {
            heap->head_ = heap->buffer_;
        } else {
            ++(heap->head_);
        }
    }

    if (node->is_min_) {
        min_heap_delete(heap, node);
    } else {
        max_heap_delete(heap, node);
    }

    if (heap->min_size_ < heap->max_size_) {
        struct node_* max_top = heap->max_heap_[0];
        max_heap_delete(heap, max_top);
        min_heap_insert(heap, max_top);
    } else if (heap->min_size_ > heap->max_size_ + 1) {
        struct node_* min_top = heap->min_heap_[0];
        min_heap_delete(heap, min_top);
        max_heap_insert(heap, min_top);
    }
}

static inline double dual_heap_get_median(struct dual_heap_* heap) {
    if (heap->min_size_ == 0 && heap->max_size_ == 0) {
        return NAN;
    } else if (heap->min_size_ == heap->max_size_) {
        return (heap->min_heap_[0]->value_ + heap->max_heap_[0]->value_) / 2.0;
    } else {
        return heap->min_heap_[0]->value_;
    }
}

static void rolling_median_float64(PyArrayObject* input, PyArrayObject* output, int window, int min_count, int axis) {
    Py_ssize_t n = PyArray_SHAPE(input)[axis];
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];

    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);

    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;
    int count = 0, i = 0;
    npy_float64 curr, prev;

    struct dual_heap_* heap = dual_heap_init(window / 2 + 1, window / 2 + 1);

    Py_BEGIN_ALLOW_THREADS;
    while (input_iter->index < input_iter->size) {
        prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);
        output_ptr = PyArray_ITER_DATA(output_iter);
        count = 0;

        dual_heap_clear(heap);

        for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            if (npy_isfinite(curr)) {
                dual_heap_push(heap, curr);
                ++count;
            }
            *((npy_float64*)output_ptr) = NPY_NAN;
        }

        for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            if (npy_isfinite(curr)) {
                dual_heap_push(heap, curr);
                ++count;
            }
            *((npy_float64*)output_ptr) = count >= min_count ? dual_heap_get_median(heap) : NPY_NAN;
        }

        for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            prev = *((npy_float64*)prev_ptr);

            if (npy_isfinite(prev)) {
                dual_heap_pop(heap);
                --count;
            }

            if (npy_isfinite(curr)) {
                dual_heap_push(heap, curr);
                ++count;
            }

            *((npy_float64*)output_ptr) = count >= min_count ? dual_heap_get_median(heap) : NPY_NAN;
        }

        PyArray_ITER_NEXT(input_iter);
        PyArray_ITER_NEXT(output_iter);
    }
    dual_heap_free(heap);
    Py_END_ALLOW_THREADS;
}

static PyObject* rolling_median(PyObject* self, PyObject* args, PyObject* kwargs) {
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
        rolling_median_float64(arr, median, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}
