import textwrap
import itertools
import numpy as np


__all__ = ['ChainedUfunc', 'create_chained_ufunc', 'get_chain',
           'create_from_doc', 'WrappedUfunc', 'Input', 'Output']

USE_UFUNC_CHAIN = True
if USE_UFUNC_CHAIN:
    from ufunc_chain import create as create_ufunc_chain, get_chain
else:
    def create_ufunc_chain(*args, **kwargs):
        return ChainedUfunc(*args, **kwargs)

    def get_chain(ufunc):
        if isinstance(ufunc, np.ufunc):
            ufunc = WrappedUfunc(ufunc)
        return ufunc.links


class ChainedUfunc(object):
    """A chain of ufuncs that are evaluated in turn.

    Parameters
    ----------
    links : list of tuples of (ufunc, list of int)
        Ufuncs to calculate, in order, with indices of where to get its
        inputs and put its outputs.  The indices to the chained ufuncs
        nin point to actual inputs, those beyond to first outputs, and
        then any temporaries.
    nin, nout, ntmp : int
        Total number of inputs, outputs and temporary arrays
    name : str
        Name of the chained ufunc.
    """
    def __init__(self, links, nin, nout, ntmp,
                 name='ufunc_chain', doc=None):
        self.links = links
        self.nin = nin
        self.nout = nout
        self.ntmp = ntmp
        self.nargs = nin+nout
        # should have something for different types.
        self.__name__ = name
        if doc is not None:
            self.__doc__ = doc

    def __eq__(self, other):
        return (type(other) is type(self) and
                self.__dict__ == other.__dict__)

    def __call__(self, *args, **kwargs):
        """Evaluate the ufunc.

        Args should contain all inputs, but outputs can be given after,
        or be in kwargs.
        """
        if len(args) != self.nin:
            if len(args) > self.nargs:
                raise TypeError("invalid number of arguments")
            if 'out' in kwargs:
                raise TypeError("got multiple values for 'out'")
            outputs = list(args[self.nin:])
            outputs += [None] * (self.nout - len(outputs))
            inputs = list(args[:self.nin])
        else:
            inputs = list(args)
            outputs = kwargs.pop('out', None)
            if outputs is None:
                outputs = [None] * self.nout
            elif not isinstance(outputs, tuple):
                outputs = [outputs]

            if len(outputs) != self.nout:
                raise ValueError("invalid number of outputs")

        inputs = [np.asanyarray(input_) for input_ in inputs]
        if any(output is None for output in outputs):
            shape = np.broadcast(*inputs).shape
            dtype = np.common_type(*inputs)
            outputs = [(np.zeros(shape, dtype) if output is None else output)
                       for output in outputs]

        if self.ntmp > 0:
            temporaries = [np.zeros_like(outputs[0])
                           for i in range(self.ntmp)]
        else:
            temporaries = []

        arrays = inputs + outputs + temporaries
        for ufunc, op_map in self.links:
            ufunc_inout = [arrays[i] for i in op_map]
            # As we work in-place, result is not needed.
            ufunc(*ufunc_inout)

        outputs = arrays[self.nin:self.nin+self.nout]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __repr__(self):
        return ("ChainedUfunc(links={links}, "
                "nin={nin}, "
                "nout={nout}, "
                "ntmp={ntmp})").format(
                    links=self.links,
                    nin=self.nin,
                    nout=self.nout,
                    ntmp=self.ntmp)


def arg_name(names, i, nin, nout):
    name = names[i]
    if name is None:
        if i < nin:
            name = '_i{}'.format(i)
        elif i < nin + nout:
            name = '_o{}'.format(i-nin)
        else:
            name = '_t{}'.format(i-nin-nout)
    return name


def create_chained_ufunc(links, nin, nout, ntmp,
                         name='ufunc_chain', names=None):
    nargs = nin + nout
    if names is None:
        names = [None] * (nargs + ntmp)
    names = [arg_name(names, i, nin, nout) for i in range(nargs + ntmp)]
    inputs = ', '.join(names[:nin])
    outputs = ', '.join(names[nin:nargs])
    doc0 = ("{name}({inputs}[, {outputs}, / [, out=({nones})]"
            ", *, where=True, casting='same_kind', order='K', "
            "dtype=None, subok=True[, signature, extobj])"
            .format(name=name,
                    inputs=inputs, outputs=outputs,
                    nones=('None,' if nout == 1 else
                           ', '.join(['None'] * nout))))
    code_lines = ["{name}({inputs}):".format(name=name, inputs=inputs)]
    code_lines.append('{} = None'.format(', '.join(names[nin:])))
    for uf, op_map in links:
        uf_in = [names[op_map[i]] for i in range(uf.nin)]
        uf_out = [names[op_map[i]] for i in range(uf.nin, uf.nargs)]
        code_lines.append("{outs} = {ufunc}({ins}, out={outs})"
                          .format(ufunc=uf.__name__,
                                  ins=', '.join(uf_in),
                                  outs=(uf_out[0] if uf.nout == 1 else
                                        ', '.join(uf_out))))
    code_lines.append('return {outputs}'.format(outputs=outputs))
    implements = ">>> def {}\n".format("\n...     ".join(code_lines))
    doc = ("{}\n\nImplements:\n\n{}"
           .format(doc0, textwrap.indent(implements, "    ")))
    return create_ufunc_chain(links, nin, nout, ntmp, name, doc)


def parse_doc(doc):
    code = doc.split("Implements:\n\n    >>> ")[-1]

    lines = [line.replace('...', '').strip() for line in code.split('\n')]
    while lines and 'def' not in lines[0]:
        lines = lines[1:]
    while lines and 'return' not in lines[-1]:
        lines = lines[:-1]
    name, inputs = (lines[0].split('def ')[1].replace('):', '')
                    .split('('))
    inputs = inputs.split(', ')
    nin = len(inputs)
    outputs = lines[-1].split('return ')[1].strip().split(', ')
    nout = len(outputs)
    temporaries = lines[1].split('= None')[0].strip().split(', ')[nout:]
    ntmp = len(temporaries)
    names = inputs + outputs + temporaries
    links = []

    for line in lines[2:-1]:
        ufunc, args = line[line.index('=')+1:].replace(')', '').split('(')
        ins, outs = args.split(', out=')
        outs.replace('(', '').replace(')', '')
        args = ins.split(', ') + outs.split(', ')
        links.append((getattr(np, ufunc.strip()),
                      [names.index(arg) for arg in args]))

    allnone = [None] * (nin + nout + ntmp)
    placeholders = [arg_name(allnone, i, nin, nout)
                    for i in range(nin + nout + ntmp)]
    names = [name if name != placeholder else None
             for (name, placeholder) in zip(names, placeholders)]
    return links, nin, nout, ntmp, name, names


def create_from_doc(doc):
    return create_chained_ufunc(*parse_doc(doc))


class WrappedUfunc(object):
    """Wraps a ufunc so it can be used to construct chains.

    Parameters
    ----------
    ufunc : ufunc-like (`~numpy.ufunc` or `ChainedUfunc`)
        Ufunc to wrap
    """
    def __init__(self, ufunc, outsel=None):
        if not isinstance(ufunc, (np.ufunc, ChainedUfunc)):
            raise TypeError("can only wrap ufuncs")

        self.ufunc = ufunc
        if outsel and ufunc.nout == 1:
            raise IndexError("scalar ufunc does not support indexing.")
        self.outsel = outsel

        doc = self.__doc__ = ufunc.__doc__
        if 'Implements:\n\n    >>> def ' in doc:
            (self.links, self.nin, self.nout, self.ntmp,
             self.__name__, self.names) = parse_doc(doc)
        else:
            if hasattr(ufunc, '__module__'):
                raise TypeError("ChainedUfunc with bad doc: {}"
                                .format(ufunc.__doc__))
            self.links = [(ufunc, list(range(ufunc.nargs)))]
            self.nin, self.nout = ufunc.nin, ufunc.nout
            self.ntmp = 0
            self.names = [None] * ufunc.nargs
            self.__name__ = ufunc.__name__
        self.nargs = self.nin + self.nout

    @property
    def arg_names(self):
        return [arg_name(self.names, i, self.nin, self.nout)
                for i in range(self.nargs+self.ntmp)]

    def __eq__(self, other):
        return (type(self) is type(other) and
                all(v == other.__dict__[k]
                    for k, v in self.__dict__.items() if k != 'ufunc'))

    def __call__(self, *args, **kwargs):
        """Evaluate the ufunc.

        All inputs should be in args, an output can be given in kwargs.
        """
        output = self.ufunc(*args, **kwargs)
        return output[self.outsel] if self.outsel else output

    def _adjusted_maps(self, offsets):
        return [[i + offsets[i] for i in link[1]]
                for link in self.links]

    @classmethod
    def from_chain(cls, links, nin, nout, ntmp,
                   name='chained_ufunc', names=None):
        ufunc = create_chained_ufunc(links, nin, nout, ntmp,
                                     name, names)
        return cls(ufunc)

    def __and__(self, other):
        if not isinstance(other, WrappedUfunc):
            return NotImplemented

        # first adjust the input and output maps for self
        self_maps = self._adjusted_maps(
            [0]*self.nin +
            [other.nin]*self.nout +
            [other.nin + other.nout]*self.ntmp)
        other_maps = other._adjusted_maps(
            [self.nin]*other.nin +
            [self.nin + self.nout]*(other.nout + other.ntmp))

        in_names = self.names[:self.nin] + other.names[:other.nin]
        out_names = (self.names[self.nin:self.nargs] +
                     other.names[other.nin:other.nargs])
        tmp_names = [(o_n if s_n is None else s_n)
                     for (s_n, o_n) in itertools.zip_longest(
                             self.names[self.nargs:],
                             other.names[other.nargs:])]
        links = [(l[0], m) for l, m in zip(self.links + other.links,
                                           self_maps + other_maps)]
        return self.from_chain(links,
                               self.nin + other.nin,
                               self.nout + other.nout,
                               max(self.ntmp, other.ntmp),
                               names=(in_names + out_names +
                                      tmp_names))

    def __or__(self, other):
        if isinstance(other, (ChainedUfunc, np.ufunc)):
            other = self.__class__(other)
        elif not isinstance(other, WrappedUfunc):
            return NotImplemented

        # First determine whether our outputs suffice for inputs;
        # if not, we need new inputs, and thus have to rearrange our maps.
        extra_nin = max(other.nin - self.nout, 0)
        extra_nout = max(self.nout - other.nin, 0)
        nin = self.nin + extra_nin
        # take as many inputs as needed and can be provided from our outputs
        # (or rather where they will be after remapping).
        n_other_in_from_self_out = min(other.nin, self.nout)

        # For the maps before the appending, we just need to add offsets
        # so that any new inputs can be accomodated. Note that some outputs
        # may become temporaries or vice versa; that's OK.
        self_offsets = ([0]*self.nin +
                        [extra_nin]*n_other_in_from_self_out +
                        [extra_nin+extra_nout]*(self.nout + self.ntmp -
                                                n_other_in_from_self_out))
        self_op_maps = self._adjusted_maps(self_offsets)

        # Now see how the number of outputs changes relative to other.
        nout = other.nout + extra_nout
        other_offsets = (
            [nin]*n_other_in_from_self_out +
            [self.nin - n_other_in_from_self_out]*extra_nin +
            [nin - other.nin]*other.nout +
            [nin - other.nin + extra_nout]*other.ntmp)

        other_op_maps = other._adjusted_maps(other_offsets)

        ntmp = max(self.nout + self.ntmp - nout,
                   other.nout + other.ntmp - nout, 0)

        # Find names for new map positions from self and other
        s_names = [None]*(nin + nout + ntmp)
        o_names = [None]*(nin + nout + ntmp)
        for i, (offset, name) in enumerate(zip(self_offsets, self.names)):
            s_names[i+offset] = name
        for i, (offset, name) in enumerate(zip(other_offsets, other.names)):
            o_names[i+offset] = name
        names = [(o_n if o_n else s_n)
                 for (o_n, s_n) in zip(o_names, s_names)]

        links = [(l[0], m) for (l, m) in zip(self.links + other.links,
                                             self_op_maps + other_op_maps)]
        return self.from_chain(links, nin, nout, ntmp, names=names)

    def _can_handle(self, ufunc, method, *inputs, **kwargs):
        can_handle = ('out' not in kwargs and method == '__call__' and
                      all(isinstance(a, WrappedUfunc) for a in inputs))
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not self._can_handle(ufunc, method, *inputs, **kwargs):
            return NotImplemented

        wrapped_ufunc = self.__class__(ufunc)

        if any(a.outsel for a in inputs):
            if not all(a.ufunc is self.ufunc for a in inputs):
                raise NotImplementedError("different selections on ufuncs")

            if [a.outsel for a in inputs] != list(range(self.ufunc.nout)):
                raise NotImplementedError("not all outputs")

            return self | wrapped_ufunc
        else:
            # combine inputs
            combined_input = inputs[0]
            for input_ in inputs[1:]:
                combined_input &= input_
            return combined_input | wrapped_ufunc

    def __getitem__(self, item):
        if self.ufunc.nout == 1:
            raise IndexError("scalar ufunc does not support indexing.")

        try:
            list(range(self.ufunc.nout))[item]
        except IndexError:
            raise IndexError("index out of range.")

        return self.__class__(self.ufunc, item)

    def __repr__(self):
        return ("WrappedUfunc({}{})"
                .format(self.ufunc,
                        ', {}'.format(self.outsel) if self.outsel else ''))

    def graph(self):
        from graphviz import Digraph
        arg_names = self.arg_names

        dg_in = Digraph('in', node_attr=dict(shape='point', rank='min'))
        for i in range(self.nin):
            dg_in.node(arg_names[i])
        dg_out = Digraph('out', node_attr=dict(shape='point', rank='max'))
        for i in range(self.nout):
            dg_out.node(arg_names[self.nin + i])

        dg = Digraph(graph_attr=dict(rankdir='LR'))
        dg.subgraph(dg_in)
        dg.subgraph(dg_out)
        array_label = ' | '.join(
            ['<{name}> {name}'.format(name=arg_names[i])
             for i in range(self.nargs+self.ntmp)])
        dg_arrays = Digraph('arrays', node_attr=dict(
            shape='record', group='arrays', label=array_label))
        for iu in range(len(self.links) + 1):
            dg_arrays.node('node{}'.format(iu))
        dg.subgraph(dg_arrays)

        # Link inputs to node0.
        node_port = "node{}:{}".format
        for i in range(self.nin):
            arg_name = arg_names[i]
            dg.edge(arg_name, node_port(0, arg_name))
        for iu, (ufunc, op_map) in enumerate(self.links):
            name = "ufunc{}".format(iu)
            # ensure array holders are aligned
            dg.edge(node_port(iu, arg_names[0]),
                    node_port(iu+1, arg_names[0]),
                    style='invis')
            # connect arrays to ufunc inputs.
            dg.node(name, label=ufunc.__name__)
            for i in range(ufunc.nin):
                arg_name = arg_names[op_map[i]]
                if ufunc.nin == 1:
                    extra = dict()
                else:
                    extra = dict(headlabel=str(i))
                dg.edge(node_port(iu, arg_name), name, **extra)
            # connect ufunc outputs to next array.
            for i in range(ufunc.nout):
                arg_name = arg_names[op_map[ufunc.nin+i]]
                if ufunc.nout == 1:
                    extra = dict()
                else:
                    extra = dict(taillabel=str(i))
                dg.edge(name, node_port(iu+1, arg_name), **extra)
        # finally, connect last array to outputs.
        for i in range(self.nin, self.nargs):
            arg_name = arg_names[i]
            dg.edge(node_port(len(self.links), arg_name), arg_name)

        return dg


class InOut(object):
    def __init__(self, name=None):
        self.name = name
        self.names = [name]


class Input(InOut):
    def _can_handle(self, method, *inputs, **kwargs):
        can_handle = (method == '__call__' and
                      all(isinstance(a, (Input, WrappedUfunc))
                          for a in inputs) and
                      'out' not in kwargs)
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # we're a mapping, and should turn the ufunc that called us
        # into a chainable version.
        if not self._can_handle(method, *inputs, **kwargs):
            return NotImplemented

        if not all(isinstance(a, Input) for a in inputs):
            if ufunc.nin > 2:
                raise NotImplementedError('>2 inputs, with some not Input')

            input_first = isinstance(inputs[0], Input)
            result = inputs[input_first]
            input_ = inputs[1-input_first]
            if result.ufunc.nout > 1:
                raise NotImplementedError('>1 output for non-Input input')

            result |= ufunc
            names = result.names
            if input_first:
                op_maps = result._adjusted_maps([1]*result.nin +
                                                [0]*(result.nout +
                                                     result.ntmp))
                op_maps[-1][ufunc.nin-1] = 0
                names[:result.nin] = ([input_.name] +
                                      result.names[:result.nin-1])
            else:
                op_maps = [l[1] for l in result.links]
                names[result.ufunc.nin - 1] = input_.name

        else:
            result = WrappedUfunc(ufunc)
            names = [a.name for a in inputs] + result.names[ufunc.nin:]
            op_maps = [l[1] for l in result.links]
            if len(names) - names.count(None) != len(set(names) - {None}):
                raise NotImplementedError("duplicate names")

        links = [(l[0], m) for (l, m) in zip(result.links, op_maps)]
        result = result.from_chain(links,
                                   result.nin, result.nout,
                                   result.ntmp, result.__name__, names)

        return result


class Output(InOut):
    def _can_handle(self, method, *inputs, **kwargs):
        can_handle = (method == '__call__' and
                      all(isinstance(a, (Input, WrappedUfunc))
                          for a in inputs) and
                      'out' in kwargs and
                      all(a is None or isinstance(a, Output)
                          for a in kwargs['out']))
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # we're a mapping, and should turn the ufunc that called us
        # into a chainable version.
        if not self._can_handle(method, *inputs, **kwargs):
            return NotImplemented

        outputs = kwargs.pop('out')
        result = getattr(ufunc, method)(*inputs, **kwargs)

        names = result.names
        # For now, only copy names
        for iout, output in zip(result.links[-1][1][ufunc.nin:],
                                outputs):
            if output is not None:
                names[iout] = output.name
        if names != result.names:
            result = result.from_chain(result.links,
                                       result.nin, result.nout, result.ntmp,
                                       result.__name__, names)
        return result
