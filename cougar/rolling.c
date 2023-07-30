#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#include "rolling_max.h"
#include "rolling_mean.h"
#include "rolling_median.h"
#include "rolling_min.h"
#include "rolling_quantile.h"
#include "rolling_rank.h"
#include "rolling_std.h"
#include "rolling_sum.h"
#include "rolling_var.h"

/* Module method table */
static PyMethodDef rolling_methods[] = {
    {"rolling_sum", (PyCFunction)rolling_sum, METH_VARARGS | METH_KEYWORDS, rolling_sum_doc},
    {"rolling_mean", (PyCFunction)rolling_mean, METH_VARARGS | METH_KEYWORDS, "Rolling mean"},
    {"rolling_std", (PyCFunction)rolling_std, METH_VARARGS | METH_KEYWORDS, "Rolling std"},
    {"rolling_var", (PyCFunction)rolling_var, METH_VARARGS | METH_KEYWORDS, "Rolling var"},
    {"rolling_max", (PyCFunction)rolling_max, METH_VARARGS | METH_KEYWORDS, "Rolling max"},
    {"rolling_min", (PyCFunction)rolling_min, METH_VARARGS | METH_KEYWORDS, "Rolling min"},
    {"rolling_median", (PyCFunction)rolling_median, METH_VARARGS | METH_KEYWORDS, "Rolling median"},
    {"rolling_rank", (PyCFunction)rolling_rank, METH_VARARGS | METH_KEYWORDS, "Rolling rank"},
    {"rolling_quantile", (PyCFunction)rolling_quantile, METH_VARARGS | METH_KEYWORDS, "Rolling quantile"},
    {NULL, NULL, 0, NULL}};

/* Module structure */
static struct PyModuleDef rolling_module = {
    PyModuleDef_HEAD_INIT,
    "rolling",      /* name of module */
    "",             /* Doc string (may be NULL) */
    -1,             /* Size of per-interpreter state or -1 */
    rolling_methods /* Method table */
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit_rolling(void) {
    PyObject* module = PyModule_Create(&rolling_module);
    import_array();
    return module;
}