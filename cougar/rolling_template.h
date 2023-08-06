#ifndef __ROLLING_TEMPLATE_H__
#define __ROLLING_TEMPLATE_H__

#define Rolling_Concat(a, b) a##_##b

#define Rolling_GetValue(name, dtype) (name = (*((dtype*)name##_ptr)))
#define Rolling_SetValue(name, value, dtype) *((dtype*)name##_ptr) = (value)

#define Rolling_Prepare()                                                     \
    Py_ssize_t n = PyArray_SHAPE(source)[axis];                               \
    Py_ssize_t source_stride = PyArray_STRIDES(source)[axis];                 \
    Py_ssize_t target_stride = PyArray_STRIDES(target)[axis];                 \
                                                                              \
    PyArrayIterObject* source_iter =                                          \
        (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)source, &axis); \
    PyArrayIterObject* target_iter =                                          \
        (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)target, &axis); \
                                                                              \
    char *target_ptr = NULL, *curr_ptr = NULL, *prev_ptr = NULL;              \
    size_t i = 0;                                                             \
    SourceType curr, prev;

#define Rolling_While while (source_iter->index < source_iter->size)

#define Rolling_InitIter()                                       \
    prev_ptr = curr_ptr = (char*)PyArray_ITER_DATA(source_iter); \
    target_ptr = (char*)PyArray_ITER_DATA(target_iter);

#define Rolling_NextIter()          \
    PyArray_ITER_NEXT(source_iter); \
    PyArray_ITER_NEXT(target_iter);

#define Rolling_ForMinCount \
    for (i = 0; i < min_count - 1; ++i, curr_ptr += source_stride, target_ptr += target_stride)
#define Rolling_ForWindow \
    for (; i < window; ++i, curr_ptr += source_stride, target_ptr += target_stride)
#define Rolling_ForN \
    for (; i < n; ++i, curr_ptr += source_stride, prev_ptr += source_stride, target_ptr += target_stride)

#endif  // __ROLLING_TEMPLATE_H__