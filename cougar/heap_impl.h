#ifdef T

#include "math.h"

#define concat(a, b) a##_##b

#define heap_node(dtype) struct concat(heap_node, dtype)

#endif