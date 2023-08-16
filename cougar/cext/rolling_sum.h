#ifndef __ROLLING_SUM_H__
#define __ROLLING_SUM_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#define Method sum

#define Rolling_Init() \
    size_t count = 0;  \
    TargetType sum = 0;

#define Rolling_Reset() \
    count = 0;          \
    sum = 0;

#define Rolling_Insert(value) \
    ++count;                  \
    sum += value;

#define Rolling_Evict(value) \
    --count;                 \
    sum -= value;

#define Rolling_InsertAndEvict(curr, prev) \
    sum += curr - prev;

#define Rolling_Compute() (sum)

#define TargetType npy_float64

#define SourceType npy_float64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_float32
#include "rolling_impl.h"
#undef SourceType

#undef Rolling_Evict
#undef Rolling_Insert
#undef Rolling_Reset
#undef Rolling_Init

#define __COUGAR_NO_VERIFY__

#define Rolling_Init() \
    TargetType sum = 0;

#define Rolling_Reset() \
    sum = 0;

#define Rolling_Insert(curr) \
    sum += curr;

#define Rolling_Evict(prev) \
    sum -= prev;

#define SourceType npy_int64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_int32
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_bool
#include "rolling_impl.h"
#undef SourceType

#undef __COUGAR_NO_VERIFY__
#undef TargetType

#undef Rolling_Compute
#undef Rolling_Init
#undef Rolling_Reset
#undef Rolling_Insert
#undef Rolling_Evict
#undef Rolling_InsertAndEvict

#undef Method

static PyObject*
rolling_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
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

    output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT64, 0);
    sum = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_sum_npy_float64(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_sum_npy_float32(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_sum_npy_int64(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_sum_npy_int32(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_BOOL) {
        rolling_sum_npy_bool(arr, sum, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

PyDoc_STRVAR(
    rolling_sum_doc,
    "rolling_sum(arr, window, min_count=-1, axis=-1)\n"
    "--\n\n"
    "Rolling sum\n\n");

#endif