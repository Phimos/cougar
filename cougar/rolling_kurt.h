#ifndef __ROLLING_KURT_H__
#define __ROLLING_KURT_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "rolling_template.h"

#define Method kurt

#define Rolling_Signature(name, dtype)                                           \
    static void Rolling_Concat(rolling_##name, dtype)(PyArrayObject * source,    \
                                                      PyArrayObject * target,    \
                                                      int window, int min_count, \
                                                      int axis, int ddof)

#define Rolling_Init()                        \
    size_t count = 0;                         \
    TargetType icount, icount2, icount3;      \
    TargetType delta, delta2, delta3, delta4; \
    TargetType mean, m2, m3, m4;

#define Rolling_Reset() \
    count = 0;          \
    mean = m2 = m3 = m4 = 0;

#define Rolling_Insert(value)                                                                                                    \
    ++count;                                                                                                                     \
    delta = value - mean;                                                                                                        \
    delta2 = delta * delta;                                                                                                      \
    delta3 = delta * delta2;                                                                                                     \
    delta4 = delta * delta3;                                                                                                     \
    icount = 1.0 / (TargetType)count;                                                                                            \
    icount2 = icount * icount;                                                                                                   \
    icount3 = icount2 * icount;                                                                                                  \
    mean += delta * icount;                                                                                                      \
    m4 += ((delta4 * (count - 1) * (icount - 3 * icount2 + 3 * icount3)) + 6 * delta2 * m2 * icount2 - 4 * delta * m3 * icount); \
    m3 += delta3 * (1 - icount) * (1 - 2 * icount) - 3 * delta * m2 * icount;                                                    \
    m2 += delta2 * (1 - icount);

#define Rolling_Evict(value)                                                                                                     \
    --count;                                                                                                                     \
    delta = value - mean;                                                                                                        \
    delta2 = delta * delta;                                                                                                      \
    delta3 = delta * delta2;                                                                                                     \
    delta4 = delta * delta3;                                                                                                     \
    icount = 1.0 / (TargetType)count;                                                                                            \
    icount2 = icount * icount;                                                                                                   \
    icount3 = icount2 * icount;                                                                                                  \
    mean -= delta * icount;                                                                                                      \
    m4 -= ((delta4 * (count + 1) * (icount + 3 * icount2 + 3 * icount3)) - 6 * delta2 * m2 * icount2 - 4 * delta * m3 * icount); \
    m3 -= delta3 * (1 + icount) * (1 + 2 * icount) - 3 * delta * m2 * icount;                                                    \
    m2 -= delta2 * (1 + icount);

#define Rolling_Assign()                                                                                                             \
    if (count >= min_count) {                                                                                                        \
        m2 = m2 < 0 ? 0 : m2;                                                                                                        \
        *((TargetType*)target_ptr) = (((count * m4) / (m2 * m2) - 3) * (count + 1) + 6) * (count - 1) / ((count - 2) * (count - 3)); \
    } else {                                                                                                                         \
        *((TargetType*)target_ptr) = NPY_NAN;                                                                                        \
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

#define __ROLLING_NO_VERIFY
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

#undef __ROLLING_NO_VERIFY
#undef TargetType

#undef Rolling_Assign
#undef Rolling_Init
#undef Rolling_Reset
#undef Rolling_Insert
#undef Rolling_Evict
#undef Rolling_Signature

#undef Method

static PyObject* rolling_kurt(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1, ddof = 0;

    static char* keywords[] = {"arr", "window", "min_count", "axis", "ddof", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|iii", keywords,
                                &input, &window, &min_count, &axis, &ddof);

    PyArrayObject *arr = NULL, *kurtosis = NULL;
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
    kurtosis = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_kurt_npy_float64(arr, kurtosis, window, min_count, axis, ddof);
    } else if (dtype == NPY_FLOAT32) {
        rolling_kurt_npy_float32(arr, kurtosis, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT64) {
        rolling_kurt_npy_int64(arr, kurtosis, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT32) {
        rolling_kurt_npy_int32(arr, kurtosis, window, min_count, axis, ddof);
    } else if (dtype == NPY_BOOL) {
        rolling_kurt_npy_bool(arr, kurtosis, window, min_count, axis, ddof);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif