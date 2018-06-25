/* -*- c -*- */

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"
#include "numpy/npy_3kcompat.h"

/*
 * Should think about deallocation:
 * point to UFUNCS and INCREF/DECREF them?
 * ufunc_dealloc:
 * - frees core_num_dims, core_offsets, core_signature, ptr, op_flags
 * - XDECREFs ->userloops, ->obj
 * ->ptr is meant for any dynamically allocated memory! (in ufunc_frompyfunc)
 */
typedef struct {
    int nin;
    int nout;
    int ntmp;
    int nufunc;
    int *nop;
    PyUFuncGenericFunction *functions;
    void **data;
    int *op_indices;
    npy_intp *steps;
} ufunc_chain_info;


static void
inner_loop_chain(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
{
    int i, iu, iop;
    int n=dimensions[0];
    char *ufunc_args[NPY_MAXARGS];
    npy_intp ufunc_steps[NPY_MAXARGS];
    ufunc_chain_info *chain_info = (ufunc_chain_info *)data;
    int *index = chain_info->op_indices;
    int ntmp=chain_info->ntmp;
    int ninout=chain_info->nin + chain_info->nout;
    char *tmp_mem=NULL;
    char **tmps={NULL};
    if (ntmp > 0) {
        npy_intp s = ntmp * sizeof(*tmps);
        for (i = 0; i < ntmp; i++) {
            s += n * chain_info->steps[i];
        }
        tmp_mem = PyArray_malloc(s);
        if (!tmp_mem) {
            PyErr_NoMemory();
            return;
        }
        tmps = (char **)tmp_mem;
        s = ntmp * sizeof(*tmps);
        for (i = 0; i < ntmp; i++) {
            tmps[i] = tmp_mem + s;
            s += n * chain_info->steps[i];
        }
    }
    for (iu = 0; iu < chain_info->nufunc; iu++) {
        int nop = chain_info->nop[iu];
        for (iop = 0; iop < nop; iop++) {
            /* printf("iu=%d, iop=%d, *index=%d\n", iu, iop, *index); */
            if (*index < ninout) {
                ufunc_args[iop] = args[*index];
                ufunc_steps[iop] = steps[*index];
            }
            else {
                ufunc_args[iop] = tmps[*index - ninout];
                ufunc_steps[iop] = chain_info->steps[*index - ninout];
            }
            index++;
        }
        chain_info->functions[iu](ufunc_args, dimensions, ufunc_steps,
                                  chain_info->data[iu]);
    }
    if (ntmp > 0) {
        PyArray_free(tmp_mem);
    }
}


static PyObject *
create_ufunc_chain(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    int nin, nout, ntmp;
    PyObject *ufunc_list=NULL, *op_maps=NULL;
    char *name=NULL, *doc=NULL;
    char *kw_list[] = {
        "ufuncs", "op_maps", "nin", "nout", "ntmp", "name", "doc", NULL};
    int nufunc, nmaps, nindices, ntypes;
    PyUFuncObject **ufuncs=NULL;
    PyUFuncObject *ufunc;
    PyObject *chained_ufunc=NULL;
    int *op_indices=NULL;
    int *ufunc_nop=NULL;
    char *types=NULL;
    PyUFuncGenericFunction *functions=NULL, *inner_loops=NULL;
    ufunc_chain_info *chain_info=NULL;
    void **data=NULL, **inner_data=NULL;
    npy_intp *chain_steps=NULL;
    static PyTypeObject *ufunc_cls=NULL;
    int iu, itype, iop;

    if (ufunc_cls == NULL) {
        PyObject *mod = PyImport_ImportModule("numpy.core");
        if (mod == NULL) {
            return NULL;
        }
        ufunc_cls = (PyTypeObject*)PyObject_GetAttrString(mod, "ufunc");
        Py_DECREF(mod);
        if (ufunc_cls == NULL) {
            return NULL;
        }
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOiii|ss", kw_list,
                                     &ufunc_list, &op_maps,
                                     &nin, &nout, &ntmp,
                                     &name, &doc)) {
        return NULL;
    }
    nufunc = PyList_Size(ufunc_list);
    if (nufunc < 0) {
        goto fail;
    }
    nmaps = PyList_Size(op_maps);
    if (nmaps < 0) {
        goto fail;
    }
    if (nmaps != nufunc) {
        PyErr_SetString(PyExc_ValueError,
                        "'op_maps' must have as many entries as 'ufuncs'");
        goto fail;
    }
    op_indices = PyArray_malloc(nufunc * (nin + nout + ntmp) *
                                sizeof(*op_indices));
    ufuncs = PyArray_malloc(nufunc * sizeof(*ufuncs));
    ufunc_nop = PyArray_malloc(nufunc * sizeof(*ufunc_nop));
    ntypes = 1;
    types = PyArray_malloc((nin + nout) * ntypes * sizeof(*types));
    chain_info = PyArray_malloc(ntypes * sizeof(*chain_info));
    functions = PyArray_malloc(ntypes * sizeof(*functions));
    inner_loops = PyArray_malloc(ntypes * nufunc * sizeof(*inner_loops));
    inner_data = PyArray_malloc(ntypes * nufunc * sizeof(*inner_data));
    data = PyArray_malloc(ntypes * sizeof(*data));
    if (ntmp > 0) {
        chain_steps = PyArray_malloc(ntypes * ntmp * sizeof(*chain_steps));
    }
    if (ufuncs == NULL || ufunc_nop == NULL || op_indices == NULL ||
        data == NULL || inner_loops == NULL || inner_data == NULL ||
        types == NULL || chain_info == NULL || functions == NULL ||
        (ntmp > 0 && chain_steps == NULL)) {
        PyErr_NoMemory();
        goto fail;
    }
    nindices = 0;
    for (iu = 0; iu < nufunc; iu++) {
        int nop;
        PyObject *op_map = PyList_GetItem(op_maps, iu);
        ufunc = (PyUFuncObject *)PyList_GetItem(ufunc_list, iu);
        if (Py_TYPE(ufunc) != ufunc_cls) {
            PyErr_SetString(PyExc_TypeError,
                            "every entry in 'ufuncs' should be a ufunc");
            goto fail;
        }
        nop = PyList_Size(op_map);
        if (nop < 0) {
            goto fail;
        }
        if (nop != ufunc->nargs) {
            PyErr_SetString(PyExc_ValueError,
                "op_map list should contain entries for each ufunc operand");
            goto fail;
        }

        ufuncs[iu] = ufunc;
        ufunc_nop[iu] = ufunc->nargs;
        for (iop = 0; iop < nop; iop++) {
            int op_index;
            PyObject *obj = PyNumber_Index(PyList_GetItem(op_map, iop));
            if (obj == NULL) {
                goto fail;
            }
            op_index = PyLong_AsLong(obj);
            Py_DECREF(obj);
            if (op_index == -1 && PyErr_Occurred()) {
                goto fail;
            }
            if (op_index < 0 || op_index >= nin + nout + ntmp) {
                PyErr_Format(PyExc_ValueError,
                             "index %d larger than maximum allowed: "
                             "nin+nout+ntmp=%d+%d+%d=%d",
                             op_index, nin, nout, ntmp, nin + nout + ntmp);
                goto fail;
            }
            op_indices[nindices++] = op_index;
        }
    }
    op_indices = PyArray_realloc(op_indices, nindices * sizeof(*op_indices));
    for (itype = 0; itype < ntypes; itype++) {
        int i, nop = nin + nout;
        for (i = itype*nop; i < (itype+1)*nop; i++) {
            types[i] = NPY_DOUBLE;
        }
        functions[itype] = (PyUFuncGenericFunction)inner_loop_chain;
        chain_info[itype].nin = nin;
        chain_info[itype].nout = nout;
        chain_info[itype].ntmp = ntmp;
        chain_info[itype].nufunc = nufunc;
        chain_info[itype].nop = ufunc_nop;
        chain_info[itype].functions = inner_loops + itype * nufunc;
        chain_info[itype].data = inner_data + itype * nufunc;
        if (ntmp > 0) {
            chain_info[itype].steps = chain_steps + itype *ntmp;;
            for (i = 0; i < ntmp; i++) {
                chain_info[itype].steps[i] = sizeof(double);
            }
        }
        else {
            chain_info[itype].steps = NULL;
        }
        for (iu = 0; iu < nufunc; iu++) {
            ufunc = ufuncs[iu];
            for (i = 0; i < ufunc->ntypes; i++) {
                int it = i * ufunc->nargs;
                if (ufunc->types[it] == NPY_DOUBLE) {
                    break;
                }
            }
            /* printf("iu=%d, i=%d\n", iu, i); */
            if (i == ufunc->ntypes) {
                PyErr_Format(PyExc_ValueError,
                             "ufunc '%s' does not support DOUBLE",
                             ufunc->name);
                goto fail;
            }
            chain_info[itype].functions[iu] = ufuncs[iu]->functions[i];
            chain_info[itype].data[iu] = ufuncs[iu]->data[i];
        }
        chain_info[itype].op_indices = op_indices;
        data[itype] = (void *)(chain_info + itype);
    }
    chained_ufunc = PyUFunc_FromFuncAndData(
        functions, data, types, ntypes,
        nin, nout, PyUFunc_None, name, doc, 0);

    PyArray_free(ufuncs);
    Py_DECREF(op_maps);
    Py_DECREF(ufunc_list);
    return chained_ufunc;

  fail:
    PyArray_free(ufuncs);
    PyArray_free(ufunc_nop);
    PyArray_free(op_indices);
    PyArray_free(data);
    PyArray_free(inner_loops);
    PyArray_free(inner_data);
    PyArray_free(types);
    PyArray_free(chain_info);
    PyArray_free(functions);
    PyArray_free(chain_steps);
    Py_XDECREF(ufunc_list);
    Py_XDECREF(op_maps);
    return NULL;
}

static PyMethodDef ufunc_chain_methods[] = {
    {"create", (PyCFunction)create_ufunc_chain, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ufunc_chain",
        NULL,
        -1,
        ufunc_chain_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_ufunc_chain(void) {
    PyObject *m;
    PyObject *d;
    PyObject *version;

    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    import_array();
    import_ufunc();

    d = PyModule_GetDict(m);

    version = PyString_FromString("0.1");
    PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);

    return m;
}
