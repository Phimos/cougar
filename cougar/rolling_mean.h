#ifndef __ROLLING_MEAN_H__
#define __ROLLING_MEAN_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#include "template.h"

#define RollingMean_Init(itype, otype) \
    Rolling_Init(itype, otype);        \
    npy_##otype sum = 0;

#define RollingMean_InitIter() \
    Rolling_InitIter();        \
    sum = 0;

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
        Rolling_While {                                         \
            RollingMean_InitIter();                             \
                                                                \
            Rolling_ForMinCount {                               \
                RollingMean_StepMinCount(itype, otype);         \
            }                                                   \
                                                                \
            Rolling_ForWindow {                                 \
                RollingMean_StepWindow(itype, otype);           \
            }                                                   \
                                                                \
            Rolling_ForN {                                      \
                RollingMean_StepN(itype, otype);                \
            }                                                   \
                                                                \
            Rolling_NextIter();                                 \
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
        Rolling_While {                                                     \
            RollingMean_InitIter();                                         \
                                                                            \
            Rolling_ForMinCount {                                           \
                RollingMean_StepMinCount_NoVerify(itype, otype);            \
            }                                                               \
                                                                            \
            Rolling_ForWindow {                                             \
                RollingMean_StepWindow_NoVerify(itype, otype);              \
            }                                                               \
                                                                            \
            Rolling_ForN {                                                  \
                RollingMean_StepN_NoVerify(itype, otype);                   \
            }                                                               \
                                                                            \
            Rolling_NextIter();                                             \
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