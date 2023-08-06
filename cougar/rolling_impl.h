#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "rolling_template.h"

#ifdef SourceType
#ifdef TargetType
#ifdef Method

#ifndef Rolling_Valid
#define __ROLLING_VALID
#define Rolling_Valid(value) (npy_isfinite(value))
#endif  // Rolling_Valid

#ifndef Rolling_Signature
#define __ROLLING_SIGNATURE
#define Rolling_Signature(name, stype, ttype)                                          \
    static void rolling_##name##_##stype(PyArrayObject* source, PyArrayObject* target, \
                                         int window, int min_count, int axis)
#endif  // Rolling_Signature

#ifndef Rolling_Init
#define __ROLLING_INIT
#define Rolling_Init(stype, ttype) ;
#endif  // Rolling_Init

#ifndef Rolling_Reset
#define __ROLLING_RESET
#define Rolling_Reset() ;
#endif  // Rolling_Reset

#ifdef Rolling_Insert
#ifdef Rolling_Evict
#ifdef Rolling_Compute

#ifndef Rolling_StepMinCount
#define __ROLLING_STEP_MIN_COUNT
#ifdef __ROLLING_NO_VERIFY
#define Rolling_StepMinCount(stype, ttype) \
    Rolling_GetValue(curr, stype);         \
    Rolling_Insert(curr);                  \
    Rolling_SetValue(target, NPY_NAN, ttype);
#else  // __ROLLING_NO_VERIFY
#define Rolling_StepMinCount(stype, ttype) \
    Rolling_GetValue(curr, stype);         \
    if (Rolling_Valid(curr)) {             \
        Rolling_Insert(curr);              \
    }                                      \
    Rolling_SetValue(target, NPY_NAN, ttype);
#endif  // __ROLLING_NO_VERIFY
#endif  // Rolling_StepMinCount

#ifndef Rolling_StepWindow
#define __ROLLING_STEP_WINDOW
#ifdef __ROLLING_NO_VERIFY
#define Rolling_StepWindow(stype, ttype) \
    Rolling_GetValue(curr, stype);       \
    Rolling_Insert(curr);                \
    Rolling_SetValue(target, Rolling_Compute(), ttype);
#else  // __ROLLING_NO_VERIFY
#define Rolling_StepWindow(stype, ttype) \
    Rolling_GetValue(curr, stype);       \
    if (Rolling_Valid(curr)) {           \
        Rolling_Insert(curr);            \
    }                                    \
    Rolling_SetValue(target, Rolling_Compute(), ttype);
#endif  // __ROLLING_NO_VERIFY
#endif  // Rolling_StepWindow

#ifndef Rolling_StepN
#define __ROLLING_STEP_N
#ifdef __ROLLING_NO_VERIFY
#define Rolling_StepN(stype, ttype) \
    Rolling_GetValue(curr, stype);  \
    Rolling_GetValue(prev, stype);  \
    Rolling_Insert(curr);           \
    Rolling_Evict(prev);            \
    Rolling_SetValue(target, Rolling_Compute(), ttype)
#else  // __ROLLING_NO_VERIFY
#define Rolling_StepN(stype, ttype) \
    Rolling_GetValue(curr, stype);  \
    Rolling_GetValue(prev, stype);  \
    if (Rolling_Valid(curr)) {      \
        Rolling_Insert(curr);       \
    }                               \
    if (Rolling_Valid(prev)) {      \
        Rolling_Evict(prev);        \
    }                               \
    Rolling_SetValue(target, Rolling_Compute(), ttype)
#endif  // __ROLLING_NO_VERIFY
#endif  // Rolling_StepN

#endif  // Rolling_Compute
#endif  // Rolling_Evict
#endif  // Rolling_Insert

#define Rolling_Main(name, stype, ttype)            \
    Rolling_Signature(name, stype, ttype) {         \
        Rolling_Prepare(stype, ttype);              \
        Rolling_Init(stype, ttype);                 \
                                                    \
        Py_BEGIN_ALLOW_THREADS;                     \
        Rolling_While {                             \
            Rolling_InitIter();                     \
            Rolling_Reset();                        \
            Rolling_ForMinCount {                   \
                Rolling_StepMinCount(stype, ttype); \
            }                                       \
            Rolling_ForWindow {                     \
                Rolling_StepWindow(stype, ttype);   \
            }                                       \
            Rolling_ForN {                          \
                Rolling_StepN(stype, ttype);        \
            }                                       \
            Rolling_NextIter();                     \
        }                                           \
        Py_END_ALLOW_THREADS;                       \
    }

Rolling_Main(Method, SourceType, TargetType)

#ifdef __ROLLING_VALID
#undef Rolling_Valid
#undef __ROLLING_VALID
#endif  // __ROLLING_VALID

#ifdef __ROLLING_SIGNATURE
#undef Rolling_Signature
#undef __ROLLING_SIGNATURE
#endif  // __ROLLING_SIGNATURE

#ifdef __ROLLING_INIT
#undef Rolling_Init
#undef __ROLLING_INIT
#endif  // __ROLLING_INIT

#ifdef __ROLLING_STEP_MIN_COUNT
#undef Rolling_StepMinCount
#undef __ROLLING_STEP_MIN_COUNT
#endif  // __ROLLING_STEP_MIN_COUNT

#ifndef __ROLLING_STEP_WINDOW
#undef Rolling_StepWindow
#undef __ROLLING_STEP_WINDOW
#endif  // __ROLLING_STEP_WINDOW

#ifdef __ROLLING_STEP_N
#undef Rolling_StepN
#undef __ROLLING_STEP_N
#endif  // __ROLLING_STEP_N

#ifdef __ROLLING_RESET
#undef Rolling_Reset
#undef __ROLLING_RESET
#endif  // __ROLLING_RESET

#endif  // Method
#endif  // TargetType
#endif  // SourceType