#ifndef __ROLLING_SUM_H__
#define __ROLLING_SUM_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#define RollingSum_IMPL(itype, otype)                                                                                   \
    static void rolling_sum_##itype(PyArrayObject* input, PyArrayObject* output, int window, int min_count, int axis) { \
        Py_ssize_t n = PyArray_SHAPE(input)[axis];                                                                      \
        Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];                                                         \
        Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];                                                       \
                                                                                                                        \
        PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);            \
        PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);          \
                                                                                                                        \
        char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;                                                    \
        int count = 0, i = 0;                                                                                           \
        npy_##itype curr, prev;                                                                                         \
        npy_##otype sum = 0;                                                                                            \
                                                                                                                        \
        Py_BEGIN_ALLOW_THREADS;                                                                                         \
        while (input_iter->index < input_iter->size) {                                                                  \
            prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);                                                        \
            output_ptr = PyArray_ITER_DATA(output_iter);                                                                \
            count = 0;                                                                                                  \
            sum = 0;                                                                                                    \
                                                                                                                        \
            for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {                \
                curr = *((npy_##itype*)curr_ptr);                                                                       \
                if (npy_isfinite(curr)) {                                                                               \
                    sum += curr;                                                                                        \
                    ++count;                                                                                            \
                }                                                                                                       \
                *((npy_##otype*)output_ptr) = NPY_NAN;                                                                  \
            }                                                                                                           \
                                                                                                                        \
            for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {                            \
                curr = *((npy_##itype*)curr_ptr);                                                                       \
                if (npy_isfinite(curr)) {                                                                               \
                    sum += curr;                                                                                        \
                    ++count;                                                                                            \
                }                                                                                                       \
                *((npy_##otype*)output_ptr) = count >= min_count ? sum : NPY_NAN;                                       \
            }                                                                                                           \
                                                                                                                        \
            for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {       \
                curr = *((npy_##itype*)curr_ptr);                                                                       \
                prev = *((npy_##itype*)prev_ptr);                                                                       \
                if (npy_isfinite(curr)) {                                                                               \
                    if (npy_isfinite(prev)) {                                                                           \
                        sum += curr - prev;                                                                             \
                    } else {                                                                                            \
                        sum += curr;                                                                                    \
                        ++count;                                                                                        \
                    }                                                                                                   \
                } else if (npy_isfinite(prev)) {                                                                        \
                    sum -= prev;                                                                                        \
                    --count;                                                                                            \
                }                                                                                                       \
                *((npy_##otype*)output_ptr) = count >= min_count ? sum : NPY_NAN;                                       \
            }                                                                                                           \
                                                                                                                        \
            PyArray_ITER_NEXT(input_iter);                                                                              \
            PyArray_ITER_NEXT(output_iter);                                                                             \
        }                                                                                                               \
                                                                                                                        \
        Py_END_ALLOW_THREADS;                                                                                           \
    }

RollingSum_IMPL(float64, float64);
RollingSum_IMPL(float32, float32);
RollingSum_IMPL(int64, float64);
RollingSum_IMPL(int32, float64);
RollingSum_IMPL(bool, float64);

static PyObject* rolling_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *sum = NULL;
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
    sum = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_sum_float64(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_sum_float32(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_sum_int64(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_sum_int32(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_BOOL) {
        rolling_sum_bool(arr, sum, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

PyDoc_STRVAR(
    rolling_sum_doc,
    "rolling_sum(arr, window, min_count=-1, axis=-1)\n");

#endif