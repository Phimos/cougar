#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "stdio.h"

#include "rolling_max.h"
#include "rolling_mean.h"
#include "rolling_median.h"
#include "rolling_sum.h"

/* Module method table */
static PyMethodDef rolling_methods[] = {
    {"rolling_sum", (PyCFunction)rolling_sum, METH_VARARGS | METH_KEYWORDS, "Rolling sum"},
    {"rolling_mean", (PyCFunction)rolling_mean, METH_VARARGS | METH_KEYWORDS, "Rolling mean"},
    {"rolling_max", (PyCFunction)rolling_max, METH_VARARGS | METH_KEYWORDS, "Rolling max"},
    {"rolling_median", (PyCFunction)rolling_median, METH_VARARGS | METH_KEYWORDS, "Rolling median"},
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