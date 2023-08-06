#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "rolling_template.h"

// #define __ROLLING_NO_VERIFY

#ifdef SourceType
#ifdef TargetType
#ifdef Method

#ifndef Rolling_Valid
#define __ROLLING_VALID
#ifdef __ROLLING_NO_VERIFY
#define Rolling_Valid(value) (1)
#else  // __ROLLING_NO_VERIFY
#define Rolling_Valid(value) (npy_isfinite(value))
#endif  // __ROLLING_NO_VERIFY
#endif  // Rolling_Valid

#ifndef Rolling_Signature
#define __ROLLING_SIGNATURE
#define Rolling_Signature(name, dtype)                                        \
    static void Rolling_Concat(rolling_##name, dtype)(PyArrayObject * source, \
                                                      PyArrayObject * target, \
                                                      int window, int min_count, int axis)
#endif  // Rolling_Signature

#ifndef Rolling_Init
#define __ROLLING_INIT
#define Rolling_Init() ;
#endif  // Rolling_Init

#ifndef Rolling_Reset
#define __ROLLING_RESET
#define Rolling_Reset() ;
#endif  // Rolling_Reset

#ifndef Rolling_Finalize
#define __ROLLING_FINALIZE
#define Rolling_Finalize() ;
#endif  // Rolling_Finalize

#ifdef Rolling_Insert
#ifdef Rolling_Evict
#ifdef Rolling_Compute

#ifndef Rolling_StepMinCount
#define __ROLLING_STEP_MIN_COUNT
#define Rolling_StepMinCount()          \
    Rolling_GetValue(curr, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    Rolling_SetValue(target, NPY_NAN, TargetType);
#endif  // Rolling_StepMinCount

#ifndef Rolling_StepWindow
#define __ROLLING_STEP_WINDOW
#define Rolling_StepWindow()            \
    Rolling_GetValue(curr, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    Rolling_SetValue(target, Rolling_Compute(), TargetType);
#endif  // Rolling_StepWindow

#ifndef Rolling_StepN
#define __ROLLING_STEP_N
#define Rolling_StepN()                 \
    Rolling_GetValue(curr, SourceType); \
    Rolling_GetValue(prev, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    if (Rolling_Valid(prev)) {          \
        Rolling_Evict(prev);            \
    }                                   \
    Rolling_SetValue(target, Rolling_Compute(), TargetType)
#endif  // Rolling_StepN

#endif  // Rolling_Compute
#endif  // Rolling_Evict
#endif  // Rolling_Insert

#define Rolling_Main(name)                \
    Rolling_Signature(name, SourceType) { \
        Rolling_Prepare();                \
        Rolling_Init();                   \
                                          \
        Py_BEGIN_ALLOW_THREADS;           \
        Rolling_While {                   \
            Rolling_InitIter();           \
            Rolling_Reset();              \
            Rolling_ForMinCount {         \
                Rolling_StepMinCount();   \
            }                             \
            Rolling_ForWindow {           \
                Rolling_StepWindow();     \
            }                             \
            Rolling_ForN {                \
                Rolling_StepN();          \
            }                             \
            Rolling_NextIter();           \
        }                                 \
        Py_END_ALLOW_THREADS;             \
        Rolling_Finalize();               \
    }

Rolling_Main(Method)

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

#ifdef __ROLLING_FINALIZE
#undef Rolling_Finalize
#undef __ROLLING_FINALIZE
#endif  // __ROLLING_FINALIZE

#endif  // Method
#endif  // TargetType
#endif  // SourceType