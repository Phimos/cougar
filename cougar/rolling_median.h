#ifndef __ROLLING_MEDIAN_H__
#define __ROLLING_MEDIAN_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "heap.h"

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

#endif