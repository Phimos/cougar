#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "rolling_template.h"

// #define __COUGAR_NO_VERIFY__

// #define __COUGAR_NO_VERIFY__ // TODO: add this to cougar.h -> control there is no need to verify
// #define __COUGAR_COUNTER__ // TODO: add this to cougar.h -> control whether to use a counter or not

#ifdef SourceType
#ifdef TargetType
#ifdef Method

#ifndef Rolling_Valid
#define __ROLLING_VALID__
#ifdef __COUGAR_NO_VERIFY__
#define Rolling_Valid(value) (1)
#else  // __COUGAR_NO_VERIFY__
#define Rolling_Valid(value) (npy_isfinite(value))
#endif  // __COUGAR_NO_VERIFY__
#endif  // Rolling_Valid

#ifndef Rolling_Signature
#define __ROLLING_SIGNATURE__
#define Rolling_Signature(name, dtype)                                        \
    static void Rolling_Concat(rolling_##name, dtype)(PyArrayObject * source, \
                                                      PyArrayObject * target, \
                                                      int window, int min_count, int axis)
#endif  // Rolling_Signature

#ifndef Rolling_Init
#define __ROLLING_INIT__
#define Rolling_Init() ;
#endif  // Rolling_Init

#ifndef Rolling_Reset
#define __ROLLING_RESET__
#define Rolling_Reset() ;
#endif  // Rolling_Reset

#ifndef Rolling_Finalize
#define __ROLLING_FINALIZE__
#define Rolling_Finalize() ;
#endif  // Rolling_Finalize

#ifdef Rolling_Compute
#ifndef Rolling_Assign
#define __ROLLING_ASSIGN__
#define Rolling_Assign() \
    Rolling_SetValue(target, Rolling_Compute(), TargetType)
#endif  // Rolling_Assign
#endif  // Rolling_Compute

#ifndef Rolling_AssignWindow
#define __ROLLING_ASSIGN_WINDOW__
#define Rolling_AssignWindow() \
    Rolling_Assign()
#endif  // Rolling_AssignWindow

#ifndef Rolling_AssignN
#define __ROLLING_ASSIGN_N__
#define Rolling_AssignN() \
    Rolling_Assign()
#endif  // Rolling_AssignN

#ifdef Rolling_Insert
#ifdef Rolling_Evict
#ifdef Rolling_Assign

#ifndef Rolling_StepMinCount
#define __ROLLING_STEP_MIN_COUNT__
#define Rolling_StepMinCount()          \
    Rolling_GetValue(curr, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    Rolling_SetValue(target, NPY_NAN, TargetType);
#endif  // Rolling_StepMinCount

#ifndef Rolling_StepWindow
#define __ROLLING_STEP_WINDOW__
#define Rolling_StepWindow()            \
    Rolling_GetValue(curr, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    Rolling_AssignWindow();
#endif  // Rolling_StepWindow

#ifndef Rolling_StepN
#define __ROLLING_STEP_N__
#define Rolling_StepN()                 \
    Rolling_GetValue(curr, SourceType); \
    Rolling_GetValue(prev, SourceType); \
    if (Rolling_Valid(curr)) {          \
        Rolling_Insert(curr);           \
    }                                   \
    if (Rolling_Valid(prev)) {          \
        Rolling_Evict(prev);            \
    }                                   \
    Rolling_AssignN();
#endif  // Rolling_StepN

#endif  // Rolling_Assign
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

#ifdef __ROLLING_VALID__
#undef Rolling_Valid
#undef __ROLLING_VALID__
#endif  // __ROLLING_VALID__

#ifdef __ROLLING_SIGNATURE__
#undef Rolling_Signature
#undef __ROLLING_SIGNATURE__
#endif  // __ROLLING_SIGNATURE__

#ifdef __ROLLING_INIT__
#undef Rolling_Init
#undef __ROLLING_INIT__
#endif  // __ROLLING_INIT__

#ifdef __ROLLING_STEP_MIN_COUNT__
#undef Rolling_StepMinCount
#undef __ROLLING_STEP_MIN_COUNT__
#endif  // __ROLLING_STEP_MIN_COUNT__

#ifndef __ROLLING_STEP_WINDOW__
#undef Rolling_StepWindow
#undef __ROLLING_STEP_WINDOW__
#endif  // __ROLLING_STEP_WINDOW__

#ifdef __ROLLING_STEP_N__
#undef Rolling_StepN
#undef __ROLLING_STEP_N__
#endif  // __ROLLING_STEP_N__

#ifdef __ROLLING_RESET__
#undef Rolling_Reset
#undef __ROLLING_RESET__
#endif  // __ROLLING_RESET__

#ifdef __ROLLING_FINALIZE__
#undef Rolling_Finalize
#undef __ROLLING_FINALIZE__
#endif  // __ROLLING_FINALIZE__

#ifdef __ROLLING_ASSIGN__
#undef Rolling_Assign
#undef __ROLLING_ASSIGN__
#endif  // __ROLLING_ASSIGN__

#ifdef __ROLLING_ASSIGN_WINDOW__
#undef Rolling_AssignWindow
#undef __ROLLING_ASSIGN_WINDOW__
#endif  // __ROLLING_ASSIGN_WINDOW__

#ifdef __ROLLING_ASSIGN_N__
#undef Rolling_AssignN
#undef __ROLLING_ASSIGN_N__
#endif  // __ROLLING_ASSIGN_N__

#endif  // Method
#endif  // TargetType
#endif  // SourceType