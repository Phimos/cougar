#ifndef __MONOTONIC_QUEUE_H__
#define __MONOTONIC_QUEUE_H__

#include "math.h"
#include "stdlib.h"

#include "debug.h"

#ifdef T
#undef T
#endif

#define T npy_float64
#include "monotonic_queue_impl.h"
#undef T

#define T npy_float32
#include "monotonic_queue_impl.h"
#undef T

#define T npy_int64
#include "monotonic_queue_impl.h"
#undef T

#define T npy_int32
#include "monotonic_queue_impl.h"
#undef T

#define monotonic_queue_concat(a, b) a##_##b
#define monotonic_queue(dtype) struct monotonic_queue_concat(monotonic_queue, dtype)
#define monotonic_queue_method(name, dtype) \
    monotonic_queue_concat(monotonic_queue_##name, dtype)

#endif