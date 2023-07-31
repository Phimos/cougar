#ifndef __ROLLING_MEAN_H__
#define __ROLLING_MEAN_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#define Rolling_GetValue(name, dtype) (name = (*((npy_##dtype*)name##_ptr)))
#define Rolling_SetValue(name, value, dtype) *((npy_##dtype*)name##_ptr) = (value)

#define RollingMean_Init(itype, otype)                                                                     \
    Py_ssize_t n = PyArray_SHAPE(input)[axis];                                                             \
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];                                                \
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];                                              \
                                                                                                           \
    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);   \
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis); \
                                                                                                           \
    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;                                           \
    int count = 0, i = 0;                                                                                  \
    npy_##itype curr, prev;                                                                                \
    npy_##otype sum = 0;

#define RollingMean_InitIter()                                  \
    prev_ptr = curr_ptr = (char*)PyArray_ITER_DATA(input_iter); \
    output_ptr = (char*)PyArray_ITER_DATA(output_iter);         \
    count = 0;                                                  \
    sum = 0;

#define RollingMean_NextIter()     \
    PyArray_ITER_NEXT(input_iter); \
    PyArray_ITER_NEXT(output_iter);

#define RollingMean_While while (input_iter->index < input_iter->size)

#define RollingMean_ForMinCount for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride)
#define RollingMean_ForWindow for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride)
#define RollingMean_ForN for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride)

#define RollingMean_Compute_NoVerify() (sum / count)
#define RollingMean_Compute() ((count >= min_count) ? RollingMean_Compute_NoVerify() : NPY_NAN)

#define RollingMean_Check(value) npy_isfinite(value)

#define RollingMean_StepMinCount_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);                      \
    sum += curr;                                        \
    ++count;                                            \
    Rolling_SetValue(output, NPY_NAN, otype);

#define RollingMean_StepMinCount(itype, otype) \
    Rolling_GetValue(curr, itype);             \
    if (RollingMean_Check(curr)) {             \
        sum += curr;                           \
        ++count;                               \
    }                                          \
    Rolling_SetValue(output, NPY_NAN, otype);

#define RollingMean_StepWindow_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);                    \
    sum += curr;                                      \
    ++count;                                          \
    Rolling_SetValue(output, RollingMean_Compute_NoVerify(), otype);

#define RollingMean_StepWindow(itype, otype) \
    Rolling_GetValue(curr, itype);           \
    if (RollingMean_Check(curr)) {           \
        sum += curr;                         \
        ++count;                             \
    }                                        \
    Rolling_SetValue(output, RollingMean_Compute(), otype);

#define RollingMean_StepN_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);               \
    Rolling_GetValue(prev, itype);               \
    sum += curr;                                 \
    sum -= prev;                                 \
    Rolling_SetValue(output, RollingMean_Compute_NoVerify(), otype);

#define RollingMean_StepN(itype, otype)   \
    Rolling_GetValue(curr, itype);        \
    Rolling_GetValue(prev, itype);        \
    if (RollingMean_Check(curr)) {        \
        if (RollingMean_Check(prev)) {    \
            sum += curr - prev;           \
        } else {                          \
            sum += curr;                  \
            ++count;                      \
        }                                 \
    } else if (RollingMean_Check(prev)) { \
        sum -= prev;                      \
        --count;                          \
    }                                     \
    Rolling_SetValue(output, RollingMean_Compute(), otype);

#define RollingMean_Impl(itype, otype)                          \
    static void rolling_mean_##itype(PyArrayObject* input,      \
                                     PyArrayObject* output,     \
                                     int window, int min_count, \
                                     int axis) {                \
        RollingMean_Init(itype, otype);                         \
                                                                \
        Py_BEGIN_ALLOW_THREADS;                                 \
        RollingMean_While {                                     \
            RollingMean_InitIter();                             \
                                                                \
            RollingMean_ForMinCount {                           \
                RollingMean_StepMinCount(itype, otype);         \
            }                                                   \
                                                                \
            RollingMean_ForWindow {                             \
                RollingMean_StepWindow(itype, otype);           \
            }                                                   \
                                                                \
            RollingMean_ForN {                                  \
                RollingMean_StepN(itype, otype);                \
            }                                                   \
                                                                \
            RollingMean_NextIter();                             \
        }                                                       \
                                                                \
        Py_END_ALLOW_THREADS;                                   \
    }

#define RollingMean_Impl_NoVerify(itype, otype)                             \
    static void rolling_mean_##itype##_no_verify(PyArrayObject* input,      \
                                                 PyArrayObject* output,     \
                                                 int window, int min_count, \
                                                 int axis) {                \
        RollingMean_Init(itype, otype);                                     \
                                                                            \
        Py_BEGIN_ALLOW_THREADS;                                             \
        RollingMean_While {                                                 \
            RollingMean_InitIter();                                         \
                                                                            \
            RollingMean_ForMinCount {                                       \
                RollingMean_StepMinCount_NoVerify(itype, otype);            \
            }                                                               \
                                                                            \
            RollingMean_ForWindow {                                         \
                RollingMean_StepWindow_NoVerify(itype, otype);              \
            }                                                               \
                                                                            \
            RollingMean_ForN {                                              \
                RollingMean_StepN_NoVerify(itype, otype);                   \
            }                                                               \
                                                                            \
            RollingMean_NextIter();                                         \
        }                                                                   \
                                                                            \
        Py_END_ALLOW_THREADS;                                               \
    }

RollingMean_Impl(float64, float64);
RollingMean_Impl(float32, float32);
RollingMean_Impl_NoVerify(int64, float64);
RollingMean_Impl_NoVerify(int32, float64);
RollingMean_Impl_NoVerify(bool, float64);

static PyObject* rolling_mean(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input = NULL, *output = NULL;
    int window, min_count = -1, axis = -1;

    static char* keywords[] = {"arr", "window", "min_count", "axis", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", keywords, &input, &window, &min_count, &axis);

    PyArrayObject *arr = NULL, *mean = NULL;
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
    mean = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_mean_float64(arr, mean, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_mean_float32(arr, mean, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_mean_int64_no_verify(arr, mean, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_mean_int32_no_verify(arr, mean, window, min_count, axis);
    } else if (dtype == NPY_BOOL) {
        rolling_mean_bool_no_verify(arr, mean, window, min_count, axis);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return NULL;
    }

    Py_DECREF(arr);
    return output;
}

#endif