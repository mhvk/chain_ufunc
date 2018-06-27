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
    int nlink;
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
    int ilink;
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
    for (ilink = 0; ilink < chain_info->nlink; ilink++) {
        int type_index = chain_info->type_indices[ilink];
        PyUFuncObject *ufunc = chain_info->ufuncs[ilink];
        PyUFuncGenericFunction function = ufunc->functions[type_index];
        void *ufunc_data = ufunc->data[type_index];
        int nop = ufunc->nargs;
        int iop;
        for (iop = 0; iop < nop; iop++) {
            /* printf("ilink=%d, iop=%d, *index=%d\n", ilink, iop, *index); */
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


static PyTypeObject *get_ufunc_cls()
{
    static PyTypeObject *ufunc_cls=NULL;
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
    return ufunc_cls;
}


static PyObject *
create_ufunc_chain(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    /* Input arguments */
    char *kw_list[] = {
        "links", "nin", "nout", "ntmp", "name", "doc", NULL};
    PyObject *links;
    int nin, nout, ntmp;
    char *name=NULL, *doc=NULL;
    /* Output */
    PyUFuncObject *chained_ufunc=NULL;
    /* Directly inferred */
    size_t name_len=-1, doc_len=-1;
    PyObject *chain=NULL;
    int nlink, nindices;
    int *op_indices;
    char* tmp_mem=NULL;
    /* Calculated */
    int ntypes;
    /* Counters */
    int ilink, itype, i;
    /* Parts for which memory will be allocated */
    char *ufunc_mem=NULL, *mem;
    npy_intp mem_size, sizes[10];
    PyUFuncGenericFunction *functions;
    void **data;
    char *types;
    PyUFuncObject **ufuncs;
    ufunc_chain_info *chain_info;
    npy_intp *tmp_steps;
    int *type_indices;
    PyTypeObject *ufunc_cls = get_ufunc_cls();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oiii|ss", kw_list,
                                     &links, &nin, &nout, &ntmp,
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
     * Interpret links argument as a sequence and create a list
     * with its own references using them; we'll keep this with the
     * chained ufunc, thus ensuring all the ufuncs stay alive.
     */
    nlink = PySequence_Size(links);
    if (nlink < 0) {
        goto fail;
    }
    chain = PyList_New(nlink);
    if (chain == NULL) {
        goto fail;
    }
    /*
     * allocate temporary memory for op_indices.
     * It gets a properly sized allocation below.
     */
    tmp_mem = PyArray_malloc(sizeof(*op_indices) * nlink *
                             (nin + nout + ntmp));
    if (tmp_mem == NULL) {
        goto fail;
    }
    /*
     * Get operand indices as flattened array
     */
    op_indices = (int *)tmp_mem;
    nindices = 0;
    for (ilink = 0; ilink < nlink; ilink++) {
        PyUFuncObject *ufunc;
        PyObject *op_map;
        int iop, nop;
        PyObject *link = PySequence_GetItem(links, ilink);
        /* Transfers reference; DECREF'd if list is DECREF'd */
        PyList_SET_ITEM(chain, ilink, link);
        if (!PyTuple_Check(link)) {
            goto fail;
        }
        if (PyTuple_Size(link) != 2) {
            PyErr_SetString(PyExc_ValueError,
                "each entry in 'links' should be a tuple with 2 elements:"
                "a ufunc and a list of operand indices");
            goto fail;
        }
        ufunc = (PyUFuncObject *)PyTuple_GET_ITEM(link, 0);
        if (Py_TYPE(ufunc) != ufunc_cls || ufunc->core_enabled ||
                ufunc->ptr != NULL || ufunc->obj != NULL) {
            PyErr_SetString(PyExc_TypeError,
                "only simply ufuncs can be used to make chains");
            goto fail;
        }
        op_map = PyTuple_GET_ITEM(link, 1);
        nop = PySequence_Size(op_map);
        if (nop < 0) {
            goto fail;
        }
        if (nop != ufunc->nargs) {
            PyErr_SetString(PyExc_ValueError,
                "op_map sequence should contain an entry for each ufunc operand");
            goto fail;
        }
        for (iop = 0; iop < nop; iop++) {
            int op_index;
            PyObject *number;
            int min_index = iop < ufunc->nin ? 0: nin;
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
            if (op_index < min_index  || op_index >= nin + nout + ntmp) {
                PyErr_Format(PyExc_ValueError,
                             "index %d outside of allowed range: "
                             "%d - %d (nin=%d, nout=%d, ntmp=%d)",
                             op_index, min_index, nin + nout + ntmp,
                             nin, nout, ntmp);
                goto fail;
            }
            op_indices[nindices++] = op_index;
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
    sizes[i++] = sizeof(*types) * ntypes * (nin + nout);
    /* the ufuncs being chained and where to get/put their operands */
    sizes[i++] = sizeof(*ufuncs) * nlink;
    sizes[i++] = sizeof(*op_indices) * nindices;
    /* for each type, information on the chain inside */
    sizes[i++] = sizeof(*chain_info) * ntypes;
    sizes[i++] = sizeof(*tmp_steps) * ntypes * ntmp;
    sizes[i++] = sizeof(*type_indices) * ntypes * nlink;
    sizes[i++] = sizeof(*name) * (name_len + 1);
    sizes[i++] = sizeof(*doc) * (doc_len + 1);
    /* calculate total size, ensuring each piece is 8-byte aligned */
    mem_size = 0;
    for (i--; i >= 0; i--) {
        sizes[i] = ((sizes[i] + 7) / 8) * 8;
        mem_size += sizes[i];
    }
    /* Actually allocate the memory */
    ufunc_mem = PyArray_malloc(mem_size);
    if (ufunc_mem == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    /* Assign appropriately */
    mem = ufunc_mem;
    i = 0;
    functions = (PyUFuncGenericFunction *)mem;
    mem += sizes[i++];
    data = (void **)mem;
    mem += sizes[i++];
    types = (char *)mem;
    mem += sizes[i++];
    ufuncs = (PyUFuncObject **)mem;
    mem += sizes[i++];
    /* Copy op_indices from temporary memory allocation */
    memcpy(mem, tmp_mem, sizeof(*op_indices) * nindices);
    op_indices = (int *)mem;
    mem += sizes[i++];
    chain_info = (ufunc_chain_info *)mem;
    mem += sizes[i++];
    tmp_steps = (npy_intp *)mem;
    mem += sizes[i++];
    type_indices = (int *)mem;
    mem += sizes[i++];
    /* For name and doc, again copy information we have */
    name = strncpy(mem, name, name_len + 1);
    mem += sizes[i++];
    doc = strncpy(mem, doc, doc_len + 1);
    /* fill ufuncs array */
    for (ilink = 0; ilink < nlink; ilink++) {
        PyObject *link = PyList_GET_ITEM(chain, ilink);
        PyUFuncObject *ufunc = (PyUFuncObject *)PyTuple_GET_ITEM(link, 0);
        ufuncs[ilink] = ufunc;
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
        chain_info[itype].nlink = nlink;
        chain_info[itype].ufuncs = ufuncs;
        chain_info[itype].type_indices = type_indices + itype * nlink;
        chain_info[itype].op_indices = op_indices;
        chain_info[itype].tmp_steps = ntmp > 0 ? tmp_steps + itype * ntmp: NULL;
        for (i = 0; i < ntmp; i++) {
            chain_info[itype].tmp_steps[i] = sizeof(double);
        }
        /* find ufunc loop with correct type, and store in type_indices */
        for (ilink = 0; ilink < nlink; ilink++) {
            PyUFuncObject *ufunc = ufuncs[ilink];
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
            chain_info[itype].type_indices[ilink] = i;
        }
    }
    chained_ufunc = (PyUFuncObject *)PyUFunc_FromFuncAndData(
        functions, data, types, ntypes,
        nin, nout, PyUFunc_None, name, doc, 0);
    /*
     * We need to keep ufunc_tuple and ufunc_mem around, as they have the
     * required information, but they should be deallocated when the ufunc
     * is deleted. Use ->obj and ->ptr for this (also used in frompyfunc).
     */
    chained_ufunc->obj = chain;
    chained_ufunc->ptr = ufunc_mem;
    PyArray_free(tmp_mem);
    return (PyObject *)chained_ufunc;

  fail:
    PyArray_free(ufunc_mem);
    PyArray_free(tmp_mem);
    Py_XDECREF(chain);
    return NULL;
}


static PyObject *
get_chain(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    char *kw_list[] = {"ufunc", NULL};
    PyObject *ufunc_obj;
    PyUFuncObject *ufunc;
    PyObject *chain=NULL, *link=NULL, *op_map=NULL, *index;
    int iop;
    PyTypeObject *ufunc_cls = get_ufunc_cls();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kw_list, &ufunc_obj)) {
        return NULL;
    }
    ufunc = (PyUFuncObject *)ufunc_obj;
    if (Py_TYPE(ufunc) != ufunc_cls || ufunc->core_enabled) {
        PyErr_SetString(PyExc_TypeError,
            "can only get chain for non-generalized ufuncs");
        goto fail;
    }
    if (ufunc->obj) {
        if (!PyList_Check(ufunc->obj) || PyList_Size(ufunc->obj) < 1 ||
            !PyTuple_Check(PyList_GET_ITEM(ufunc->obj, 0))) {
            PyErr_SetString(PyExc_ValueError,
                            "ufunc does not contain chain list with "
                            "tuple elements.");
            goto fail;
        }
        chain = ufunc->obj;
        Py_INCREF(chain);
    }
    else {
        /* simple ufunc, create 1-element chain */
        chain = PyList_New(1);
        link = PyTuple_New(2);
        op_map = PyList_New(ufunc->nargs);
        if (link == NULL || chain == NULL || op_map == NULL) {
            goto fail;
        }
        for (iop = 0; iop < ufunc->nargs; iop++) {
            index = PyLong_FromLong(iop);
            PyList_SET_ITEM(op_map, iop, index);
        }
        Py_INCREF(ufunc_obj);
        PyTuple_SET_ITEM(link, 0, ufunc_obj);
        PyTuple_SET_ITEM(link, 1, op_map);
        PyList_SET_ITEM(chain, 0, link);
        /* chain is new ref, all others put inside */
    }
    return chain;

fail:
    Py_XDECREF(chain);
    Py_XDECREF(link);
    Py_XDECREF(op_map);
    return NULL;
}


static PyMethodDef ufunc_chain_methods[] = {
    {"create", (PyCFunction)create_ufunc_chain, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_chain", (PyCFunction)get_chain, METH_VARARGS | METH_KEYWORDS, NULL},
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
