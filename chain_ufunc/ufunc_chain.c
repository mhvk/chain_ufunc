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
    npy_intp ntot = dimensions[0];
    char *ufunc_args[NPY_MAXARGS];
    npy_intp ufunc_steps[NPY_MAXARGS];
    int has0_only[NPY_MAXARGS] = {0};
    double singletons[NPY_MAXARGS];
    ufunc_chain_info *chain_info = (ufunc_chain_info *)data;
    int ntmp = chain_info->ntmp;
    int ninout = chain_info->nin + chain_info->nout;
    npy_intp *tmp_steps = chain_info->tmp_steps;
    char *tmp_mem = NULL;
    char **tmps = {NULL};
    int ilink, iop;
    npy_intp bufsize = ntot > 8192? 8192: ntot;   /* somehow get actual bufsize!? */
    if (ntmp > 0) {
        int i;
        npy_intp s = ntmp * sizeof(*tmps);
        for (i = 0; i < ntmp; i++) {
            s += bufsize * chain_info->tmp_steps[i];
        }
        tmp_mem = PyArray_malloc(s);
        if (!tmp_mem) {
            PyErr_NoMemory();
            return;
        }
        tmps = (char **)tmp_mem;
        for (--i; i >= 0; --i) {
            s -= bufsize * chain_info->tmp_steps[i];
            tmps[i] = tmp_mem + s;
        }
    }
    for (npy_intp offset = 0; offset < ntot; offset += bufsize) {
        int *indices = chain_info->op_indices;
        npy_intp n = ntot - offset;
        if (n > bufsize) {
            n = bufsize;  /* chunk to be dealt with */
        }
        for (ilink = 0; ilink < chain_info->nlink; ilink++) {
            int type_index = chain_info->type_indices[ilink];
            PyUFuncObject *ufunc = chain_info->ufuncs[ilink];
            PyUFuncGenericFunction function = ufunc->functions[type_index];
            void *ufunc_data = ufunc->data[type_index];
            int nin = ufunc->nin;
            int nop = ufunc->nargs;
            char *arg;
            npy_intp step;
            npy_intp dim = 1;
            for (iop = 0; iop < nop; iop++) {
                int index = indices[iop];
                /*
                 * Somewhat complicated logic to catch the case where
                 * all inputs are broadcast, i.e., have step 0.
                 */
                if (iop >= nin) {
                    has0_only[index] = (dim != n);
                }
                if (has0_only[index]) {
                    arg = (char *)(singletons + index);
                    step = 0;
                }
                else {
                    if (index < ninout) {
                        step = steps[index];
                        arg = args[index] + offset * step;
                    }
                    else {
                        arg = tmps[index - ninout];
                        step = tmp_steps[index - ninout];
                    }
                    if (step && iop < nin) {
                        dim = n;
                    }
                }
                ufunc_args[iop] = arg;
                ufunc_steps[iop] = step;
#ifdef CHAIN_DEBUG
                printf("ilink=%d, nin=%d, nout=%d, iop=%d, index=%d, has0only=%d, ",
                       ilink, nin, ufunc->nout, iop, index, has0_only[index]);
                printf("ufunc_arg=%p, ufunc_step=%ld, dim=%ld\n",
                       ufunc_args[iop], ufunc_steps[iop], dim);
#endif
            }
            function(ufunc_args, &dim, ufunc_steps, ufunc_data);
            indices += nop;
        }
        for (iop = chain_info->nin; iop < ninout; iop++) {
#ifdef CHAIN_DEBUG
            printf("final iop=%d, has0_only=%d, steps=%ld\n",
                   iop, has0_only[iop], steps[iop]);
#endif
            if (has0_only[iop] && steps[iop]) {
                /* copy missing results */
                int i;
                double c = singletons[iop];
                npy_intp step = steps[iop];
                char *tmp = args[iop] + offset * step;
                for (i = 0; i < n; i++, tmp += step) {
                    *(double *)tmp = c;
                }
            }
        }
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
    PyObject *ufunc_list=NULL;
    int nlink, nindices;
    /* Calculated */
    int ntypes;
    /* Counters */
    int ilink, itype, i;
    /* Parts for which memory will be allocated */
    char *ufunc_mem=NULL, *mem;
    npy_intp mem_size, sizes[9];
    PyUFuncGenericFunction *functions;
    void **data;
    char *types;
    PyUFuncObject **ufuncs;
    int *op_indices;
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
     * Get number of links and create a list to hold references to the ufuncs.
     * This will be our array of ufuncs, and by keeping the list with the
     * chained ufunc, we also ensure they stay alive.
     */
    nlink = PyList_Size(links);
    if (nlink < 0) {
        goto fail;
    }
    ufunc_list = PyList_New(nlink);
    if (ufunc_list == NULL) {
        goto fail;
    }
    /*
     * Get ufuncs as well as the operand indices as flattened array.
     */
    nindices = 0;
    for (ilink = 0; ilink < nlink; ilink++) {
        PyObject *ufunc_obj, *op_map;
        PyUFuncObject *ufunc;
        int nop;
        PyObject *link = PyList_GET_ITEM(links, ilink);
        if (!PyTuple_Check(link)) {
            goto fail;
        }
        if (PyTuple_Size(link) != 2) {
            PyErr_SetString(PyExc_ValueError,
                "each entry in 'links' should be a tuple with 2 elements:"
                "a ufunc and a list of operand indices");
            goto fail;
        }
        ufunc_obj = PyTuple_GET_ITEM(link, 0);
        ufunc = (PyUFuncObject *)ufunc_obj;
        if (Py_TYPE(ufunc_obj) != ufunc_cls || ufunc->core_enabled ||
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
        else if (nop != ufunc->nargs) {
            PyErr_SetString(PyExc_ValueError,
                "op_map sequence should contain an entry for each ufunc operand");
            goto fail;
        }
        Py_INCREF(ufunc_obj);
        PyList_SET_ITEM(ufunc_list, ilink, ufunc_obj);
        nindices += nop;
    }
    /* Get ufuncs as array */
    ufuncs = (PyUFuncObject **)PySequence_Fast_ITEMS(ufunc_list);
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
    sizes[i++] = sizeof(*name) * (name_len + 1);
    sizes[i++] = sizeof(*doc) * (doc_len + 1);
    /* Chain information: where to get/put ufunc operands */
    sizes[i++] = sizeof(*op_indices) * nindices;
    /* for each type, information for the loops inside */
    sizes[i++] = sizeof(*chain_info) * ntypes;
    sizes[i++] = sizeof(*tmp_steps) * ntypes * ntmp;
    sizes[i++] = sizeof(*type_indices) * ntypes * nlink;
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
    /* Basic information for the new chained ufunc itself */
    functions = (PyUFuncGenericFunction *)mem;
    mem += sizes[i++];
    data = (void **)mem;
    mem += sizes[i++];
    types = (char *)mem;
    mem += sizes[i++];
    /* For name and doc, copy information we have */
    name = strncpy(mem, name, name_len + 1);
    mem += sizes[i++];
    doc = strncpy(mem, doc, doc_len + 1);
    mem += sizes[i++];
    /* Chain information */
    op_indices = (int *)mem;
    mem += sizes[i++];
    chain_info = (ufunc_chain_info *)mem;
    mem += sizes[i++];
    tmp_steps = (npy_intp *)mem;
    mem += sizes[i++];
    type_indices = (int *)mem;
    /*
     * Fill operand indices array.
     */
    nindices = 0;
    for (ilink = 0; ilink < nlink; ilink++) {
        int iop;
        PyUFuncObject *ufunc = ufuncs[ilink];
        PyObject *link = PyList_GET_ITEM(links, ilink);
        PyObject *op_map = PyTuple_GET_ITEM(link, 1);
        for (iop = 0; iop < ufunc->nargs; iop++) {
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
    chained_ufunc->obj = ufunc_list;
    chained_ufunc->ptr = ufunc_mem;
    return (PyObject *)chained_ufunc;

  fail:
    PyArray_free(ufunc_mem);
    Py_XDECREF(ufunc_list);
    return NULL;
}


static PyObject *
get_chain(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    char *kw_list[] = {"ufunc", NULL};
    PyObject *chained_ufunc_obj;
    PyUFuncObject *chained_ufunc;
    ufunc_chain_info *chain_info;
    int nlink;
    PyUFuncObject *ufunc_array[1];
    PyUFuncObject **ufuncs = ufunc_array;
    int *op_indices = NULL;
    int ilink;
    PyObject *links=NULL, *link=NULL, *op_map=NULL;
    PyTypeObject *ufunc_cls = get_ufunc_cls();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kw_list,
                                     &chained_ufunc_obj)) {
        return NULL;
    }
    chained_ufunc = (PyUFuncObject *)chained_ufunc_obj;
    if (Py_TYPE(chained_ufunc_obj) != ufunc_cls ||
            chained_ufunc->core_enabled) {
        PyErr_SetString(PyExc_TypeError,
            "can only get chain for non-generalized ufuncs");
        return NULL;
    }
    if (chained_ufunc->obj) {
        if (!PyList_Check(chained_ufunc->obj)) {
            PyErr_SetString(PyExc_ValueError,
                            "ufunc does not contain chain list with "
                            "tuple elements.");
            return NULL;
        }
        chain_info = (ufunc_chain_info *)chained_ufunc->data[0];
        nlink = chain_info->nlink;
        ufuncs = chain_info->ufuncs;
        op_indices = chain_info->op_indices;
    }
    else {
        /* simple ufunc, create 1-element chain */
        nlink = 1;
        ufuncs[0] = chained_ufunc;
    }
    links = PyList_New(nlink);
    if (links == NULL) {
        return NULL;
    }
    for (ilink = 0; ilink < nlink; ilink++) {
        int iop;
        PyUFuncObject *ufunc = ufuncs[ilink];
        PyObject *ufunc_obj = (PyObject *)ufunc;
        int nop = ufunc->nargs;
        link = PyTuple_New(2);
        op_map = PyList_New(nop);
        if (link == NULL || op_map == NULL) {
            goto fail;
        }
        for (iop = 0; iop < nop; iop++) {
            int index = op_indices ? *op_indices++: iop;
            PyObject *index_obj = PyLong_FromLong(index);
            if (index_obj == NULL) {
                goto fail;
            }
            PyList_SET_ITEM(op_map, iop, index_obj);
        }
        Py_INCREF(ufunc_obj);
        PyTuple_SET_ITEM(link, 0, ufunc_obj);
        PyTuple_SET_ITEM(link, 1, op_map);
        PyList_SET_ITEM(links, ilink, link);
    }
    return links;

  fail:
    Py_DECREF(links);
    Py_XDECREF(op_map);
    Py_XDECREF(link);
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

    version = PyUnicode_FromString("0.1");
    PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);

    return m;
}
