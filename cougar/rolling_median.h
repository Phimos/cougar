#ifndef __ROLLING_MEDIAN_H__
#define __ROLLING_MEDIAN_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "median_heap.h"

#define Method median

#define Rolling_Init()              \
    size_t count = 0;               \
    median_heap(SourceType)* heap = \
        median_heap_method(init, SourceType)(window / 2 + 1, window / 2 + 1);

#define Rolling_Reset() \
    count = 0;          \
    median_heap_method(reset, SourceType)(heap);

#define Rolling_Insert(value) \
    ++count;                  \
    median_heap_method(push, SourceType)(heap, curr);

#define Rolling_Evict(value) \
    --count;                 \
    median_heap_method(pop, SourceType)(heap);

#define Rolling_Compute() (median_heap_method(query_median, SourceType)(heap))

#define Rolling_Finalize() \
    median_heap_method(free, SourceType)(heap);

#define TargetType npy_float64

#define SourceType npy_float64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_float32
#include "rolling_impl.h"
#undef SourceType

#define __COUGAR_NO_VERIFY__

#define SourceType npy_int64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_int32
#include "rolling_impl.h"
#undef SourceType

#undef __COUGAR_NO_VERIFY__
#undef TargetType

#undef Rolling_Finalize
#undef Rolling_Compute
#undef Rolling_Evict
#undef Rolling_Insert
#undef Rolling_Reset
#undef Rolling_Init

#undef Method

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

    output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT64, 0);
    median = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_median_npy_float64(arr, median, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_median_npy_float32(arr, median, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_median_npy_int64(arr, median, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_median_npy_int32(arr, median, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif