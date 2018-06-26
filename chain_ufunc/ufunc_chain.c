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
    PyUFuncObject **ufuncs;
    int *type_indices;
    int *op_indices;
    npy_intp *tmp_steps;
} ufunc_chain_info;


static void
inner_loop_chain(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
{
    int n = dimensions[0];
    char *ufunc_args[NPY_MAXARGS];
    npy_intp ufunc_steps[NPY_MAXARGS];
    ufunc_chain_info *chain_info = (ufunc_chain_info *)data;
    int *index = chain_info->op_indices;
    int ntmp = chain_info->ntmp;
    int ninout = chain_info->nin + chain_info->nout;
    char *tmp_mem = NULL;
    char **tmps = {NULL};
    int iu;
    if (ntmp > 0) {
        int i;
        npy_intp s = ntmp * sizeof(*tmps);
        for (i = 0; i < ntmp; i++) {
            s += n * chain_info->tmp_steps[i];
        }
        tmp_mem = PyArray_malloc(s);
        if (!tmp_mem) {
            PyErr_NoMemory();
            return;
        }
        tmps = (char **)tmp_mem;
        for (--i; i >= 0; --i) {
            s -= n * chain_info->tmp_steps[i];
            tmps[i] = tmp_mem + s;
        }
    }
    for (iu = 0; iu < chain_info->nufunc; iu++) {
        int type_index = chain_info->type_indices[iu];
        PyUFuncObject *ufunc = chain_info->ufuncs[iu];
        PyUFuncGenericFunction function = ufunc->functions[type_index];
        void *ufunc_data = ufunc->data[type_index];
        int nop = ufunc->nargs;
        int iop;
        for (iop = 0; iop < nop; iop++) {
            /* printf("iu=%d, iop=%d, *index=%d\n", iu, iop, *index); */
            if (*index < ninout) {
                ufunc_args[iop] = args[*index];
                ufunc_steps[iop] = steps[*index];
            }
            else {
                ufunc_args[iop] = tmps[*index - ninout];
                ufunc_steps[iop] = chain_info->tmp_steps[*index - ninout];
            }
            index++;
        }
        function(ufunc_args, dimensions, ufunc_steps, ufunc_data);
    }
    if (ntmp > 0) {
        PyArray_free(tmp_mem);
    }
}


static PyObject *
create_ufunc_chain(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    static PyTypeObject *ufunc_cls=NULL;
    /* Input arguments */
    char *kw_list[] = {
        "ufuncs", "op_maps", "nin", "nout", "ntmp", "name", "doc", NULL};
    PyObject *ufuncs_arg, *op_maps_arg;
    int nin, nout, ntmp;
    /* Output */
    PyUFuncObject *chained_ufunc=NULL;
    /* Directly inferred */
    char *name=NULL, *doc=NULL;
    size_t name_len=-1, doc_len=-1;
    PyObject *ufunc_tuple=NULL, *op_map_list=NULL;
    int nufunc;
    PyUFuncObject **ufuncs;
    PyObject **op_maps;
    /* Calculated */
    int ntypes, nindices;
    /* Counters */
    int iu, itype, i;
    /* Parts for which memory will be allocated */
    char *mem_ptr=NULL, *mem;
    npy_intp mem_size, sizes[9];
    PyUFuncGenericFunction *functions;
    void **data;
    char *types;
    char *name_copy;
    char *doc_copy;
    ufunc_chain_info *chain_info;
    npy_intp *tmp_steps;
    int *type_indices;
    int *op_indices;

    if (ufunc_cls == NULL) {
        PyObject *mod = PyImport_ImportModule("numpy");
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
                                     &ufuncs_arg, &op_maps_arg,
                                     &nin, &nout, &ntmp,
                                     &name, &doc)) {
        return NULL;
    }
    /* Interpret name & doc arguments */
    if (name) {
        name_len = strlen(name);
    }
    else {
        name = "ufunc_chain";
        name_len = 11;
    }
    if (doc) {
        doc_len = strlen(doc);
    }
    else {
        doc = "";
        doc_len = 0;
    }
    /*
     * Interpret ufuncs argument as a sequence and create a new tuple
     * with its own references using them; we'll keep this with the
     * chained ufunc, thus ensuring all the ufuncs stay alive.
     */
    nufunc = PySequence_Size(ufuncs_arg);
    if (nufunc < 0) {
        goto fail;
    }
    ufunc_tuple = PyTuple_New(nufunc);
    if (ufunc_tuple == NULL) {
        goto fail;
    }
    for (iu = 0; iu < nufunc; iu++) {
        PyObject *ufunc_obj = PySequence_GetItem(ufuncs_arg, iu);
        PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
        /* Uses reference from GetItem above; DECREF'd if tuple is DECREF'd */
        PyTuple_SetItem(ufunc_tuple, iu, ufunc_obj);
        if (Py_TYPE(ufunc) != ufunc_cls || ufunc->core_enabled ||
                ufunc->ptr != NULL || ufunc->obj != NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "every entry in 'ufuncs' should be a simple ufunc");
            goto fail;
        }
    }
    ufuncs = (PyUFuncObject **)PySequence_Fast_ITEMS(ufunc_tuple);
    /*
     * Check consistency with number of maps, and number of arguments inside.
     */
    op_map_list = PySequence_Fast(op_maps_arg,
                                  "'op_maps' should be a sequence");
    if (op_map_list == NULL) {
        goto fail;
    }
    if (PySequence_Fast_GET_SIZE(op_map_list) != nufunc) {
        PyErr_SetString(PyExc_ValueError,
                        "'op_maps' must have as many entries as 'ufuncs'");
        goto fail;
    }
    op_maps = PySequence_Fast_ITEMS(op_map_list);
    for (iu = 0; iu < nufunc; iu++) {
        PyObject *op_map = op_maps[iu];
        int nop = PySequence_Size(op_map);
        if (nop < 0) {
            goto fail;
        }
        if (nop != ufuncs[iu]->nargs) {
            PyErr_SetString(PyExc_ValueError,
                "op_map sequence should contain entries for each ufunc operand");
            goto fail;
        }
    }
    /*
     * Here, there should be a proper routine determine the number
     * of possible independent types.  For now, just NPY_DOUBLE.
     */
    ntypes = 1;
    /*
     * Get memory requirements for all parts that we should keep.
     */
    i = 0;
    /* basic information for the new chained ufunc itself */
    sizes[i++] = sizeof(*functions) * ntypes;
    sizes[i++] = sizeof(*data) * ntypes;
    sizes[i++] = sizeof(*types) * (nin + nout) * ntypes;
    sizes[i++] = sizeof(*name_copy) * (name_len + 1);
    sizes[i++] = sizeof(*doc_copy) * (doc_len + 1);
    /* for each type, information on the chain inside */
    sizes[i++] = sizeof(*chain_info) * ntypes;
    sizes[i++] = sizeof(*tmp_steps) * ntypes * ntmp;
    sizes[i++] = sizeof(*type_indices) * ntypes * nufunc;
    /* where to get/put ufunc operands (type-independent) */
    /* Note: overallocates, but it's not too much memory */
    sizes[i++] = sizeof(*op_indices) * nufunc * (nin + nout + ntmp);
    /* calculate total size, ensuring each piece is 8-byte aligned */
    mem_size = 0;
    for (i--; i >= 0; i--) {
        sizes[i] = ((sizes[i] + 7) / 8) * 8;
        mem_size += sizes[i];
    }
    /* Actually allocate the memory */
    mem_ptr = PyArray_malloc(mem_size);
    if (mem_ptr == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    /* Assign appropriately */
    mem = mem_ptr;
    i = 0;
    functions = (PyUFuncGenericFunction *)mem;
    mem += sizes[i++];
    data = (void **)mem;
    mem += sizes[i++];
    types = (char *)mem;
    mem += sizes[i++];
    name_copy = (char *)mem;
    mem += sizes[i++];
    doc_copy = (char *)mem;
    mem += sizes[i++];
    chain_info = (ufunc_chain_info *)mem;
    mem += sizes[i++];
    tmp_steps = (npy_intp *)mem;
    mem += sizes[i++];
    type_indices = (int *)mem;
    mem += sizes[i++];
    op_indices = (int *)mem;

    strncpy(name_copy, name, name_len + 1);
    strncpy(doc_copy, doc, doc_len + 1);
    /*
     * Get operand indices as flattened array
     */
    nindices = 0;
    for (iu = 0; iu < nufunc; iu++) {
        PyUFuncObject *ufunc = ufuncs[iu];
        PyObject *op_map = op_maps[iu];
        int nop = ufunc->nargs;
        int iop;
        for (iop = 0; iop < nop; iop++) {
            int op_index;
            PyObject *number;
            PyObject *obj = PySequence_GetItem(op_map, iop);
            if (obj == NULL) {
                goto fail;
            }
            number = PyNumber_Index(obj);
            Py_DECREF(obj);
            if (number == NULL) {
                goto fail;
            }
            op_index = PyLong_AsLong(number);
            Py_DECREF(number);
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
    /*
     * Set up ufunc information for each type (just DOUBLE for now).
     */
    for (itype = 0; itype < ntypes; itype++) {
        int i, nop=nin + nout;
        for (i = itype*nop; i < (itype+1)*nop; i++) {
            types[i] = NPY_DOUBLE;
        }
        functions[itype] = (PyUFuncGenericFunction)inner_loop_chain;
        /* data allows us to give the inner loop access to chain_info */
        data[itype] = (void *)(chain_info + itype);
        /* Fill the chain information */
        chain_info[itype].nin = nin;
        chain_info[itype].nout = nout;
        chain_info[itype].ntmp = ntmp;
        chain_info[itype].nufunc = nufunc;
        chain_info[itype].ufuncs = ufuncs;
        chain_info[itype].type_indices = type_indices + itype * nufunc;
        chain_info[itype].op_indices = op_indices;
        chain_info[itype].tmp_steps = ntmp > 0 ? tmp_steps + itype * ntmp: NULL;
        for (i = 0; i < ntmp; i++) {
            chain_info[itype].tmp_steps[i] = sizeof(double);
        }
        /* find ufunc loop with correct type, and store in type_indices */
        for (iu = 0; iu < nufunc; iu++) {
            PyUFuncObject *ufunc = ufuncs[iu];
            for (i = 0; i < ufunc->ntypes; i++) {
                int it = i * ufunc->nargs;
                if (ufunc->types[it] == NPY_DOUBLE) {
                    break;
                }
            }
            if (i == ufunc->ntypes) {
                PyErr_Format(PyExc_ValueError,
                             "ufunc '%s' does not support DOUBLE",
                             ufunc->name);
                goto fail;
            }
            chain_info[itype].type_indices[iu] = i;
        }
    }
    chained_ufunc = (PyUFuncObject *)PyUFunc_FromFuncAndData(
        functions, data, types, ntypes,
        nin, nout, PyUFunc_None, name_copy, doc, 0);
    /*
     * We need to keep ufunc_tuple and mem_ptr around, as they have the
     * required information, but they should be deallocated when the ufunc
     * is deleted. Use ->obj and ->ptr for this (also used in frompyfunc).
     */
    chained_ufunc->obj = ufunc_tuple;
    chained_ufunc->ptr = mem_ptr;
    Py_DECREF(op_map_list);
    Py_DECREF(op_maps_arg);
    Py_DECREF(ufuncs_arg);
    ufuncs = (PyUFuncObject **)PySequence_Fast_ITEMS(chained_ufunc->obj);
    return (PyObject *)chained_ufunc;

  fail:
    PyArray_free(mem_ptr);
    Py_XDECREF(ufunc_tuple);
    Py_XDECREF(ufuncs_arg);
    Py_XDECREF(op_map_list);
    Py_XDECREF(op_maps_arg);
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
