#ifndef __ROLLING_RANK_H__
#define __ROLLING_RANK_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "treap.h"

#define Method rank

#define Rolling_Init()         \
    size_t count = 0;          \
    treap(SourceType)* treap = \
        treap_method(init, SourceType)(window + 1);

#define Rolling_Reset() \
    count = 0;          \
    treap_method(reset, SourceType)(treap);

#define Rolling_Insert(value) \
    ++count;                  \
    treap_method(insert, SourceType)(treap, curr);

#define Rolling_Evict(value) \
    --count;                 \
    treap_method(remove, SourceType)(treap);

#define Rolling_Compute() \
    (count > 1 ? ((treap_method(query_rank, SourceType)(treap) - 1.0) / ((double)count - 1.0) * 2.0 - 1.0) : 0.0)

#define Rolling_Finalize() \
    treap_method(free, SourceType)(treap);

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

#undef Rolling_Compute
#undef Rolling_Init
#undef Rolling_Reset
#undef Rolling_Insert
#undef Rolling_Evict
#undef Rolling_Finalize

#undef Method

static PyObject* rolling_rank(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *rank = NULL;
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
    rank = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_rank_npy_float64(arr, rank, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_rank_npy_float32(arr, rank, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_rank_npy_int64(arr, rank, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_rank_npy_int32(arr, rank, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif