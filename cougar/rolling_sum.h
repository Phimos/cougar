#ifndef __ROLLING_SUM_H__
#define __ROLLING_SUM_H__

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#include "template.h"

#define RollingSum_Init(itype, otype) \
    Rolling_Init(itype, otype);       \
    npy_##otype sum = 0;

#define RollingSum_InitIter() \
    Rolling_InitIter();       \
    sum = 0;

#define RollingSum_Compute_NoVerify() (sum)
#define RollingSum_Compute() ((count >= min_count) ? RollingSum_Compute_NoVerify() : NPY_NAN)

#define RollingSum_Check(value) npy_isfinite(value)

#define RollingSum_StepMinCount_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);                     \
    sum += curr;                                       \
    ++count;                                           \
    Rolling_SetValue(output, NPY_NAN, otype);

#define RollingSum_StepMinCount(itype, otype) \
    Rolling_GetValue(curr, itype);            \
    if (RollingSum_Check(curr)) {             \
        sum += curr;                          \
        ++count;                              \
    }                                         \
    Rolling_SetValue(output, NPY_NAN, otype);

#define RollingSum_StepWindow_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);                   \
    sum += curr;                                     \
    ++count;                                         \
    Rolling_SetValue(output, RollingSum_Compute_NoVerify(), otype);

#define RollingSum_StepWindow(itype, otype) \
    Rolling_GetValue(curr, itype);          \
    if (RollingSum_Check(curr)) {           \
        sum += curr;                        \
        ++count;                            \
    }                                       \
    Rolling_SetValue(output, RollingSum_Compute(), otype);

#define RollingSum_StepN_NoVerify(itype, otype) \
    Rolling_GetValue(curr, itype);              \
    Rolling_GetValue(prev, itype);              \
    sum += curr;                                \
    sum -= prev;                                \
    Rolling_SetValue(output, RollingSum_Compute_NoVerify(), otype);

#define RollingSum_StepN(itype, otype)   \
    Rolling_GetValue(curr, itype);       \
    Rolling_GetValue(prev, itype);       \
    if (RollingSum_Check(curr)) {        \
        if (RollingSum_Check(prev)) {    \
            sum += curr - prev;          \
        } else {                         \
            sum += curr;                 \
            ++count;                     \
        }                                \
    } else if (RollingSum_Check(prev)) { \
        sum -= prev;                     \
        --count;                         \
    }                                    \
    Rolling_SetValue(output, RollingSum_Compute(), otype);

#define RollingSum_Impl(itype, otype)                          \
    static void rolling_sum_##itype(PyArrayObject* input,      \
                                    PyArrayObject* output,     \
                                    int window, int min_count, \
                                    int axis) {                \
        RollingSum_Init(itype, otype);                         \
                                                               \
        Py_BEGIN_ALLOW_THREADS;                                \
        Rolling_While {                                        \
            RollingSum_InitIter();                             \
                                                               \
            Rolling_ForMinCount {                              \
                RollingSum_StepMinCount(itype, otype);         \
            }                                                  \
                                                               \
            Rolling_ForWindow {                                \
                RollingSum_StepWindow(itype, otype);           \
            }                                                  \
                                                               \
            Rolling_ForN {                                     \
                RollingSum_StepN(itype, otype);                \
            }                                                  \
                                                               \
            Rolling_NextIter();                                \
        }                                                      \
                                                               \
        Py_END_ALLOW_THREADS;                                  \
    }

#define RollingSum_Impl_NoVerify(itype, otype)                         \
    static void rolling_sum_##itype##_no_verify(PyArrayObject* input,  \
                                                PyArrayObject* output, \
                                                size_t window,         \
                                                size_t min_count,      \
                                                int axis) {            \
        RollingSum_Init(itype, otype);                                 \
                                                                       \
        Py_BEGIN_ALLOW_THREADS;                                        \
        Rolling_While {                                                \
            RollingSum_InitIter();                                     \
                                                                       \
            Rolling_ForMinCount {                                      \
                RollingSum_StepMinCount_NoVerify(itype, otype);        \
            }                                                          \
                                                                       \
            Rolling_ForWindow {                                        \
                RollingSum_StepWindow_NoVerify(itype, otype);          \
            }                                                          \
                                                                       \
            Rolling_ForN {                                             \
                RollingSum_StepN_NoVerify(itype, otype);               \
            }                                                          \
                                                                       \
            Rolling_NextIter();                                        \
        }                                                              \
                                                                       \
        Py_END_ALLOW_THREADS;                                          \
    }

RollingSum_Impl(float64, float64);
RollingSum_Impl(float32, float32);
RollingSum_Impl_NoVerify(int64, float64);
RollingSum_Impl_NoVerify(int32, float64);
RollingSum_Impl_NoVerify(bool, float64);

static PyObject* rolling_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
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

    if (dtype == NPY_FLOAT32)
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT32, 0);
    else
        output = PyArray_EMPTY(PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FLOAT64, 0);
    sum = (PyArrayObject*)output;

    if (dtype == NPY_FLOAT64) {
        rolling_sum_float64(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_FLOAT32) {
        rolling_sum_float32(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT64) {
        rolling_sum_int64_no_verify(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_INT32) {
        rolling_sum_int32_no_verify(arr, sum, window, min_count, axis);
    } else if (dtype == NPY_BOOL) {
        rolling_sum_bool_no_verify(arr, sum, window, min_count, axis);
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