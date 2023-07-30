#ifndef __ROLLING_SKEW_H__
#define __ROLLING_SKEW_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#define RollingSkew_IMPL(itype, otype)                                                                                                      \
    static void rolling_skew_##itype(PyArrayObject* input, PyArrayObject* output,                                                           \
                                     int window, int min_count, int axis, int ddof) {                                                       \
        Py_ssize_t n = PyArray_SHAPE(input)[axis];                                                                                          \
        Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];                                                                             \
        Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];                                                                           \
                                                                                                                                            \
        PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);                                \
        PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);                              \
                                                                                                                                            \
        char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;                                                                        \
        int count = 0, i = 0;                                                                                                               \
        npy_##itype curr, prev;                                                                                                             \
        npy_##otype delta, mean, m2, m3;                                                                                                    \
                                                                                                                                            \
        Py_BEGIN_ALLOW_THREADS;                                                                                                             \
        while (input_iter->index < input_iter->size) {                                                                                      \
            prev_ptr = curr_ptr = (char*)PyArray_ITER_DATA(input_iter);                                                                     \
            output_ptr = (char*)PyArray_ITER_DATA(output_iter);                                                                             \
            count = 0;                                                                                                                      \
            mean = m2 = m3 = 0;                                                                                                             \
                                                                                                                                            \
            for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {                                    \
                curr = *((npy_##itype*)curr_ptr);                                                                                           \
                if (npy_isfinite(curr)) {                                                                                                   \
                    ++count;                                                                                                                \
                    delta = curr - mean;                                                                                                    \
                    mean += delta / count;                                                                                                  \
                    m3 += delta * delta * delta * (count - 1) * (count - 2) / count / count - 3 * delta * m2 / count;                       \
                    m2 += delta * delta * (count - 1) / count;                                                                              \
                }                                                                                                                           \
                *((npy_##otype*)output_ptr) = NPY_NAN;                                                                                      \
            }                                                                                                                               \
                                                                                                                                            \
            for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {                                                \
                curr = *((npy_##itype*)curr_ptr);                                                                                           \
                if (npy_isfinite(curr)) {                                                                                                   \
                    ++count;                                                                                                                \
                    delta = curr - mean;                                                                                                    \
                    mean += delta / count;                                                                                                  \
                    m3 += delta * delta * delta * (count - 1) * (count - 2) / count / count - 3 * delta * m2 / count;                       \
                    m2 += delta * delta * (count - 1) / count;                                                                              \
                }                                                                                                                           \
                if (count >= min_count) {                                                                                                   \
                    m2 = m2 < 0 ? 0 : m2;                                                                                                   \
                    *((npy_##otype*)output_ptr) = m3 * npy_sqrt(count) / (m2 * npy_sqrt(m2)) * npy_sqrt(count * (count - 1)) / (count - 2); \
                } else {                                                                                                                    \
                    *((npy_##otype*)output_ptr) = NPY_NAN;                                                                                  \
                }                                                                                                                           \
            }                                                                                                                               \
                                                                                                                                            \
            for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {                           \
                curr = *((npy_##itype*)curr_ptr);                                                                                           \
                prev = *((npy_##itype*)prev_ptr);                                                                                           \
                if (npy_isfinite(prev)) {                                                                                                   \
                    --count;                                                                                                                \
                    delta = prev - mean;                                                                                                    \
                    mean -= delta / count;                                                                                                  \
                    m3 -= delta * delta * delta * (count + 1) * (count + 2) / count / count - 3 * delta * m2 / count;                       \
                    m2 -= delta * delta * (count + 1) / count;                                                                              \
                }                                                                                                                           \
                if (npy_isfinite(curr)) {                                                                                                   \
                    ++count;                                                                                                                \
                    delta = curr - mean;                                                                                                    \
                    mean += delta / count;                                                                                                  \
                    m3 += delta * delta * delta * (count - 1) * (count - 2) / count / count - 3 * delta * m2 / count;                       \
                    m2 += delta * delta * (count - 1) / count;                                                                              \
                }                                                                                                                           \
                if (count >= min_count) {                                                                                                   \
                    m2 = m2 < 0 ? 0 : m2;                                                                                                   \
                    *((npy_##otype*)output_ptr) = m3 * npy_sqrt(count) / (m2 * npy_sqrt(m2)) * npy_sqrt(count * (count - 1)) / (count - 2); \
                } else {                                                                                                                    \
                    *((npy_##otype*)output_ptr) = NPY_NAN;                                                                                  \
                }                                                                                                                           \
            }                                                                                                                               \
                                                                                                                                            \
            PyArray_ITER_NEXT(input_iter);                                                                                                  \
            PyArray_ITER_NEXT(output_iter);                                                                                                 \
        }                                                                                                                                   \
                                                                                                                                            \
        Py_END_ALLOW_THREADS;                                                                                                               \
    }

RollingSkew_IMPL(float64, float64);
RollingSkew_IMPL(float32, float32);
RollingSkew_IMPL(int64, float64);
RollingSkew_IMPL(int32, float64);
RollingSkew_IMPL(bool, float64);

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

    if (dtype == NPY_FLOAT32)
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT32, 0);
    else
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT64, 0);
    skewness = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_skew_float64(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_FLOAT32) {
        rolling_skew_float32(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT64) {
        rolling_skew_int64(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_INT32) {
        rolling_skew_int32(arr, skewness, window, min_count, axis, ddof);
    } else if (dtype == NPY_BOOL) {
        rolling_skew_bool(arr, skewness, window, min_count, axis, ddof);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif