#ifndef __TREAP_H__
#define __TREAP_H__

#include "numpy/npy_common.h"

#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "debug.h"

#ifdef T
#undef T
#endif

#define T npy_float64
#include "treap_impl.h"
#undef T

#define T npy_float32
#include "treap_impl.h"
#undef T

#define T npy_int64
#include "treap_impl.h"
#undef T

#define T npy_int32
#include "treap_impl.h"
#undef T

#define treap_concat(a, b) a##_##b
#define treap(dtype) struct treap_concat(treap, dtype)
#define treap_method(name, dtype) treap_concat(treap_##name, dtype)

#endif