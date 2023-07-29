#ifndef __ROLLING_MAX_H__
#define __ROLLING_MAX_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#define MQ_RESET(name, index, value)                 \
    name##_head = name##_tail = name##_buffer_start; \
    name##_head->index_ = index;                     \
    name##_head->value_ = value;

#define MQ_INIT(name, size, type, default_value)                                        \
    struct item_ {                                                                      \
        type value_;                                                                    \
        int index_;                                                                     \
    };                                                                                  \
    struct item_* name##_buffer = (struct item_*)malloc(sizeof(struct item_) * (size)); \
    struct item_* name##_buffer_start = name##_buffer;                                  \
    struct item_* name##_buffer_end = name##_buffer + size;                             \
    struct item_ *name##_head, *name##_tail;                                            \
    MQ_RESET(name, -1, default_value)

#define MQ_PUSH_BACK(name, index, value)   \
    ++name##_tail;                         \
    if (name##_tail == name##_buffer_end)  \
        name##_tail = name##_buffer_start; \
    name##_tail->index_ = index;           \
    name##_tail->value_ = value;

#define MQ_POP_FRONT(name)                \
    ++name##_head;                        \
    if (name##_head == name##_buffer_end) \
        name##_head = name##_buffer_start;

#define MQ_POP_BACK(name)                   \
    if (name##_tail == name##_buffer_start) \
        name##_tail = name##_buffer_end;    \
    --name##_tail;

#define MQ_FRONT_VALUE(name) (name##_head->value_)
#define MQ_FRONT_INDEX(name) (name##_head->index_)
#define MQ_BACK_VALUE(name) (name##_tail->value_)
#define MQ_BACK_INDEX(name) (name##_tail->index_)

#define MQ_FREE(name) free(name##_buffer);

static void rolling_max_float64(PyArrayObject* input, PyArrayObject* output, int window, int min_count, int axis) {
    Py_ssize_t n = PyArray_SHAPE(input)[axis];
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];

    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);

    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;
    int count = 0, i = 0;
    npy_float64 curr, prev;

    MQ_INIT(queue, window + 1, npy_float64, -NPY_INFINITY)

    Py_BEGIN_ALLOW_THREADS;
    while (input_iter->index < input_iter->size) {
        prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);
        output_ptr = PyArray_ITER_DATA(output_iter);
        count = 0;

        MQ_RESET(queue, -1, -NPY_INFINITY)

        for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = -NPY_INFINITY;

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }

            *((npy_float64*)output_ptr) = NPY_NAN;
        }

        for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = -NPY_INFINITY;

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }
            *((npy_float64*)output_ptr) = count >= min_count ? MQ_FRONT_VALUE(queue) : NPY_NAN;
        }

        for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_float64*)curr_ptr);
            prev = *((npy_float64*)prev_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = -NPY_INFINITY;

            if (npy_isfinite(prev))
                --count;

            if (MQ_FRONT_INDEX(queue) <= i - window) {
                MQ_POP_FRONT(queue)
            }

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }

            *((npy_float64*)output_ptr) = count >= min_count ? MQ_FRONT_VALUE(queue) : NPY_NAN;
        }

        PyArray_ITER_NEXT(input_iter);
        PyArray_ITER_NEXT(output_iter);
    }

    Py_END_ALLOW_THREADS;

    MQ_FREE(queue)
}

static void rolling_max_int32(PyArrayObject* input, PyArrayObject* output, int window, int min_count, int axis) {
    Py_ssize_t n = PyArray_SHAPE(input)[axis];
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];

    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);

    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;
    int count = 0, i = 0;
    npy_int32 curr, prev;

    MQ_INIT(queue, window + 1, npy_int32, NPY_MIN_INT32)

    Py_BEGIN_ALLOW_THREADS;
    while (input_iter->index < input_iter->size) {
        prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);
        output_ptr = PyArray_ITER_DATA(output_iter);
        count = 0;

        MQ_RESET(queue, -1, NPY_MIN_INT32)

        for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_int32*)curr_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = NPY_MIN_INT32;

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }

            *((npy_float64*)output_ptr) = NPY_NAN;
        }

        for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_int32*)curr_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = NPY_MIN_INT32;

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }
            *((npy_float64*)output_ptr) = count >= min_count ? (npy_float64)MQ_FRONT_VALUE(queue) : NPY_NAN;
        }

        for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {
            curr = *((npy_int32*)curr_ptr);
            prev = *((npy_int32*)prev_ptr);

            if (npy_isfinite(curr))
                ++count;
            else
                curr = NPY_MIN_INT32;

            if (npy_isfinite(prev))
                --count;

            if (MQ_FRONT_INDEX(queue) <= i - window) {
                MQ_POP_FRONT(queue)
            }

            if (MQ_FRONT_VALUE(queue) <= curr) {
                MQ_RESET(queue, i, curr)
            } else {
                while (MQ_BACK_VALUE(queue) <= curr) {
                    MQ_POP_BACK(queue)
                }
                MQ_PUSH_BACK(queue, i, curr)
            }

            *((npy_float64*)output_ptr) = count >= min_count ? (npy_float64)MQ_FRONT_VALUE(queue) : NPY_NAN;
        }

        PyArray_ITER_NEXT(input_iter);
        PyArray_ITER_NEXT(output_iter);
    }

    Py_END_ALLOW_THREADS;

    MQ_FREE(queue)
}

PyObject* rolling_max(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *max = NULL;
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

    max = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_max_float64(arr, max, window, min_count, axis);
    }
    // else if (dtype == NPY_FLOAT32) {
    //     rolling_mean_float32(arr, sum, window, min_count, axis);
    // } else if (dtype == NPY_INT64) {
    //     rolling_mean_int64(arr, sum, window, min_count, axis);
    // }
    else if (dtype == NPY_INT32) {
        rolling_max_int32(arr, max, window, min_count, axis);
    }
    // else if (dtype == NPY_BOOL) {
    //     rolling_mean_bool(arr, sum, window, min_count, axis);
    // }
    else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif