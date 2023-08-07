#ifndef __ROLLING_QUANTILE_H__
#define __ROLLING_QUANTILE_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "rolling_template.h"

#include "treap.h"

#define Method quantile

#define Rolling_Signature(name, dtype)                                        \
    static void Rolling_Concat(rolling_##name, dtype)(PyArrayObject * source, \
                                                      PyArrayObject * target, \
                                                      int window, double q,   \
                                                      int min_count, int axis)

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
    ((count >= min_count) ? (treap_method(query_quantile, SourceType)(treap, q)) : NPY_NAN)

#define Rolling_Finalize() \
    treap_method(free, SourceType)(treap);

#define TargetType npy_float64

#define SourceType npy_float64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_float32
#include "rolling_impl.h"
#undef SourceType

#define __ROLLING_NO_VERIFY

#define SourceType npy_int64
#include "rolling_impl.h"
#undef SourceType

#define SourceType npy_int32
#include "rolling_impl.h"
#undef SourceType

#undef __ROLLING_NO_VERIFY
#undef TargetType

#undef Rolling_Signature
#undef Rolling_Compute
#undef Rolling_Init
#undef Rolling_Reset
#undef Rolling_Insert
#undef Rolling_Evict
#undef Rolling_Finalize

#undef Method

// static void rolling_quantile_float64(PyArrayObject* input, PyArrayObject* output, int window, double q, int min_count, int axis) {
//     Py_ssize_t n = PyArray_SHAPE(input)[axis];
//     Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];
//     Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];

//     PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);
//     PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis);

//     char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;
//     int count = 0, i = 0;
//     npy_float64 curr, prev;

//     struct treap_* treap = treap_init(window + 1);

//     Py_BEGIN_ALLOW_THREADS;
//     while (input_iter->index < input_iter->size) {
//         prev_ptr = curr_ptr = PyArray_ITER_DATA(input_iter);
//         output_ptr = PyArray_ITER_DATA(output_iter);
//         count = 0;

//         treap_reset(treap);

//         for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
//             curr = *((npy_float64*)curr_ptr);
//             if (npy_isfinite(curr)) {
//                 treap_insert(treap, curr);
//                 ++count;
//             }
//             *((npy_float64*)output_ptr) = NPY_NAN;
//         }

//         for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride) {
//             curr = *((npy_float64*)curr_ptr);
//             if (npy_isfinite(curr)) {
//                 treap_insert(treap, curr);
//                 ++count;
//             }
//             *((npy_float64*)output_ptr) = count >= min_count ? treap_query_quantile(treap, q) : NPY_NAN;
//         }

//         for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride) {
//             curr = *((npy_float64*)curr_ptr);
//             prev = *((npy_float64*)prev_ptr);

//             if (npy_isfinite(prev)) {
//                 treap_remove(treap);
//                 --count;
//             }

//             if (npy_isfinite(curr)) {
//                 treap_insert(treap, curr);
//                 ++count;
//             }

//             *((npy_float64*)output_ptr) = count >= min_count ? treap_query_quantile(treap, q) : NPY_NAN;
//         }

//         PyArray_ITER_NEXT(input_iter);
//         PyArray_ITER_NEXT(output_iter);
//     }
//     treap_free(treap);
//     Py_END_ALLOW_THREADS;
// }

static PyObject* rolling_quantile(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;
    double q = 0.5;

    static char* keywords[] = {"arr", "window", "quantile", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|dii", keywords, &input, &window, &q, &min_count, &axis);

    PyArrayObject *arr = NULL, *quantile = NULL;
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
    quantile = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_quantile_npy_float64(arr, quantile, window, q, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_quantile_npy_float32(arr, quantile, window, q, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_quantile_npy_int64(arr, quantile, window, q, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_quantile_npy_int32(arr, quantile, window, q, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif