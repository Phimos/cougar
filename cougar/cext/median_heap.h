#ifndef __MEDIAN_HEAP_H__
#define __MEDIAN_HEAP_H__

#include "numpy/npy_common.h"

#include "math.h"
#include "stdlib.h"

#include "debug.h"

#ifdef T
#undef T
#endif

#define T npy_float64
#include "median_heap_impl.h"
#undef T

#define T npy_float32
#include "median_heap_impl.h"
#undef T

#define T npy_int64
#include "median_heap_impl.h"
#undef T

#define T npy_int32
#include "median_heap_impl.h"
#undef T

#define median_heap_concat(a, b) a##_##b
#define median_heap(dtype) struct median_heap_concat(median_heap, dtype)
#define median_heap_method(name, dtype) \
    median_heap_concat(median_heap_##name, dtype)

#endif