#ifndef __ROLLING_MIN_H__
#define __ROLLING_MIN_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#include "monotonic_queue.h"

#define Method min

#define Rolling_Init()                   \
    size_t count = 0;                    \
    monotonic_queue(SourceType)* queue = \
        monotonic_queue_method(init, SourceType)(window, 1);

#define Rolling_Reset() \
    count = 0;          \
    monotonic_queue_method(reset, SourceType)(queue);

#define Rolling_Insert(value) \
    ++count;                  \
    monotonic_queue_method(push, SourceType)(queue, value, i);

#define Rolling_Evict(value)                                                    \
    --count;                                                                    \
    if (monotonic_queue_method(front_index, SourceType)(queue) + window <= i) { \
        monotonic_queue_method(pop_front, SourceType)(queue);                   \
    }

#define Rolling_Compute() (monotonic_queue_method(front_value, SourceType)(queue))

#define Rolling_Finalize() \
    monotonic_queue_method(free, SourceType)(queue);

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

PyObject* rolling_min(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *min = NULL;
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
    min = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_min_npy_float64(arr, min, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_min_npy_float32(arr, min, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_min_npy_int64(arr, min, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_min_npy_int32(arr, min, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif