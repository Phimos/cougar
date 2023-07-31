#ifndef __TEMPLATE_H__
#define __TEMPLATE_H__

#define Rolling_GetValue(name, dtype) (name = (*((npy_##dtype*)name##_ptr)))
#define Rolling_SetValue(name, value, dtype) *((npy_##dtype*)name##_ptr) = (value)

#define Rolling_Init(itype, otype)                                                                         \
    Py_ssize_t n = PyArray_SHAPE(input)[axis];                                                             \
    Py_ssize_t input_stride = PyArray_STRIDES(input)[axis];                                                \
    Py_ssize_t output_stride = PyArray_STRIDES(output)[axis];                                              \
                                                                                                           \
    PyArrayIterObject* input_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)input, &axis);   \
    PyArrayIterObject* output_iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)output, &axis); \
                                                                                                           \
    char *output_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;                                           \
    size_t count = 0, i = 0;                                                                               \
    npy_##itype curr, prev;

#define Rolling_While while (input_iter->index < input_iter->size)

#define Rolling_InitIter()                                      \
    prev_ptr = curr_ptr = (char*)PyArray_ITER_DATA(input_iter); \
    output_ptr = (char*)PyArray_ITER_DATA(output_iter);         \
    count = 0;

#define Rolling_NextIter()         \
    PyArray_ITER_NEXT(input_iter); \
    PyArray_ITER_NEXT(output_iter);

#define Rolling_ForMinCount for (i = 0; i < min_count - 1; ++i, curr_ptr += input_stride, output_ptr += output_stride)
#define Rolling_ForWindow for (; i < window; ++i, curr_ptr += input_stride, output_ptr += output_stride)
#define Rolling_ForN for (; i < n; ++i, curr_ptr += input_stride, prev_ptr += input_stride, output_ptr += output_stride)

#endif