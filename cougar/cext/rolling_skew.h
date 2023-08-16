#ifndef __ROLLING_SKEW_H__
#define __ROLLING_SKEW_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "rolling_template.h"

#define Method skew

#define Rolling_Signature(name, dtype)                                                 \
    static void Rolling_Concat(rolling_##name, dtype)(PyArrayObject * source,          \
                                                      PyArrayObject * target,          \
                                                      size_t window, size_t min_count, \
                                                      int axis, int ddof)

#define Rolling_Init()                \
    size_t count = 0;                 \
    TargetType icount;                \
    TargetType delta, delta2, delta3; \
    TargetType mean, m2, m3;

#define Rolling_Reset() \
    count = 0;          \
    mean = m2 = m3 = 0;

#define Rolling_Insert(value)                                                 \
    ++count;                                                                  \
    delta = value - mean;                                                     \
    delta2 = delta * delta;                                                   \
    delta3 = delta * delta2;                                                  \
    icount = 1.0 / (TargetType)count;                                         \
    mean += delta * icount;                                                   \
    m3 += delta3 * (1 - icount) * (1 - 2 * icount) - 3 * delta * m2 * icount; \
    m2 += delta2 * (1 - icount);

#define Rolling_Evict(value)                                                  \
    --count;                                                                  \
    delta = value - mean;                                                     \
    delta2 = delta * delta;                                                   \
    delta3 = delta * delta2;                                                  \
    icount = 1.0 / (TargetType)count;                                         \
    mean -= delta * icount;                                                   \
    m3 -= delta3 * (1 + icount) * (1 + 2 * icount) - 3 * delta * m2 * icount; \
    m2 -= delta2 * (1 + icount);

#define Rolling_Assign()                                                                                                                         \
    if (count >= min_count) {                                                                                                                    \
        m2 = m2 < 0 ? 0 : m2;                                                                                                                    \
        *((TargetType*)target_ptr) = m3 * npy_sqrt((double)count) / (m2 * npy_sqrt(m2)) * npy_sqrt((double)(count * (count - 1))) / (count - 2); \
    } else {                                                                                                                                     \
        *((TargetType*)target_ptr) = NPY_NAN;                                                                                                    \
    }

#define SourceType npy_float64
#define TargetType npy_float64

#include "rolling_impl.h"

#undef SourceType
#undef TargetType

#define SourceType npy_float32
#define TargetType npy_float64

#include "rolling_impl.h"

#undef SourceType
#undef TargetType

#define __COUGAR_NO_VERIFY__
#define TargetType npy_float64

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

#undef Rolling_Assign
#undef Rolling_Init
#undef Rolling_Reset
#undef Rolling_Insert
#undef Rolling_Evict
#undef Rolling_Signature

#undef Method

static PyObject* rolling_skew(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1, ddof = 0;

    static char* keywords[] = {"arr", "window", "min_count", "axis", "ddof", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|iii", keywords,
                                &input, &window, &min_count, &axis, &ddof);

    PyArrayObject *arr = NULL, *skewness = NULL;
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
    skewness = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_skew_npy_float64(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_FLOAT32) {
        rolling_skew_npy_float32(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT64) {
        rolling_skew_npy_int64(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT32) {
        rolling_skew_npy_int32(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_BOOL) {
        rolling_skew_npy_bool(arr, skewness, window, min_count, axis, ddof);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif