#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "stdio.h"

#define DEBUG 0

#if DEBUG
#define debug(fmt, ...)                               \
    do {                                              \
        fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                __LINE__, __func__, ##__VA_ARGS__);   \
    } while (0)
#else
#define debug(fmt, ...)
#endif

#endif