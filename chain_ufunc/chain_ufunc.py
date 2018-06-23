import numpy as np


__all__ = ['ChainedUfunc', 'WrappedUfunc', 'Input', 'Output']


class ChainedUfunc(object):
    """A chain of ufuncs that are evaluated in turn.

    Parameters
    ----------
    ufuncs : list of ufuc
        Ufuncs to calculate, in order.
    op_maps : list of list of int
        For each ufunc, indices of where to get its inputs and put its
        outputs.  The indices to the chained ufuncs nin point to actual
        inputs, those beyond to first outputs, and then any temporaries.
    nin, nout, ntmp : int
        total number of inputs, outputs and temporary arrays
    names : list of str, optional
        Argument names ("(in|out|tmp)<i>" by default).
    """
    def __init__(self, ufuncs, op_maps, nin, nout, ntmp, names=None):
        self.ufuncs = ufuncs
        self.op_maps = op_maps
        self.nin = nin
        self.nout = nout
        self.ntmp = ntmp
        self.nargs = nin+nout
        # should have something for different types.
        if names is None:
            names = [None] * (nin + nout + ntmp)
        self.names = names

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
        for ufunc, op_map in zip(self.ufuncs, self.op_maps):
            ufunc_inout = [arrays[i] for i in op_map]
            # As we work in-place, result is not needed.
            ufunc(*ufunc_inout)

        outputs = arrays[self.nin:self.nin+self.nout]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __repr__(self):
        if all(name is None for name in self.names):
            names = ""
        else:
            names = "names={}".format(self.names)
        return ("ChainedUfunc(ufuncs={ufuncs}, "
                "op_maps={op_maps}, "
                "nin={nin}, "
                "nout={nout}, "
                "ntmp={ntmp}, "
                "{names})").format(
                    ufuncs=self.ufuncs,
                    op_maps=self.op_maps,
                    nin=self.nin,
                    nout=self.nout,
                    ntmp=self.ntmp,
                    names=names)


class WrappedUfunc(object):
    """Wraps a ufunc so it can be used to construct chains.

    Parameters
    ----------
    ufunc : ufunc-like (`~numpy.ufunc` or `ChainedUfunc`)
        Ufunc to wrap
    """
    def __init__(self, ufunc, outsel=None):
        if isinstance(ufunc, np.ufunc):
            op_map = list(range(ufunc.nargs))
            ufunc = ChainedUfunc([ufunc], [op_map], ufunc.nin, ufunc.nout, 0)
        elif not isinstance(ufunc, ChainedUfunc):
            raise TypeError("can only wrap ufuncs")

        self.ufunc = ufunc

        if outsel and ufunc.nout == 1:
            raise IndexError("scalar ufunc does not support indexing.")
        self.outsel = outsel

        for attr in ('ufuncs', 'nin', 'nout', 'nargs', 'ntmp',
                     'op_maps', 'names'):
            setattr(self, attr, getattr(self.ufunc, attr))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.__dict__ == other.__dict__)

    def __call__(self, *args, **kwargs):
        """Evaluate the ufunc.

        All inputs should be in args, an output can be given in kwargs.
        """
        output = self.ufunc(*args, **kwargs)
        return output[self.outsel] if self.outsel else output

    def _adjusted_maps(self, off_in, off_out, off_tmp):
        if off_in == 0 and off_out == 0 and (off_tmp == 0 or self.ntmp == 0):
            return self.op_maps

        nin = self.nin
        ninplusout = nin + self.nout
        old_maps = self.op_maps
        new_maps = []
        for old_map in old_maps:
            new_map = []
            for i in old_map:
                if i < nin:
                    i += off_in
                elif i < ninplusout:
                    i += off_out
                else:
                    i += off_tmp
                new_map.append(i)
            new_maps.append(new_map)
        return new_maps

    def __and__(self, other):
        if not isinstance(other, WrappedUfunc):
            return NotImplemented

        # first adjust the input and output maps for self
        self_maps = self._adjusted_maps(0, other.nin, other.nin + other.nout)
        other_maps = other._adjusted_maps(self.nin, self.nin + self.nout,
                                          self.nin + self.nout)
        names = (self.names[:self.nin] +
                 other.names[:other.nin] +
                 self.names[self.nin:self.nargs] +
                 other.names[other.nin:other.nargs] +
                 self.names[self.nargs:] +
                 other.names[other.nargs:])
        return self.__class__(ChainedUfunc(
            self.ufuncs + other.ufuncs,
            self_maps + other_maps,
            self.nin + other.nin, self.nout + other.nout,
            max(self.ntmp, other.ntmp),
            names))

    def __or__(self, other):
        if isinstance(other, (ChainedUfunc, np.ufunc)):
            other = self.__class__(other)
        elif not isinstance(other, WrappedUfunc):
            return NotImplemented

        cls = type(self)
        # First determine whether our outputs suffice for inputs;
        # if not, we need new inputs, and thus have to rearrange our maps.
        extra_nin = max(other.nin - self.nout, 0)
        nin = self.nin + extra_nin
        # take as many inputs as needed and can be provided from our outputs
        # (or rather where they will be after remapping).
        n_other_in_from_self_out = min(other.nin, self.nout)
        # For now, these inputs are only allowed at the start of other
        # (need to assign a temporary for them otherwise)
        for other_op_map, other_ufunc in zip(other.op_maps[1:],
                                             other.ufuncs[1:]):
            if any(i < n_other_in_from_self_out
                   for i in other_op_map[:other_ufunc.nin]):
                raise NotImplementedError(
                    "Cannot yet append chain in which an input does not "
                    "immediately use the outputs.")

        other_input_remap = list(range(nin, nin+n_other_in_from_self_out))
        if extra_nin:
            # add any missing ones from the new inputs.
            other_input_remap += list(range(self.nin, nin))

        # For the maps before the appending, we just need to add offsets
        # so that any new inputs can be accomodated. Note that some outputs
        # may become temporaries or vice versa; that's OK.
        self_op_maps = self._adjusted_maps(0, extra_nin, extra_nin)

        # Now see how the number of outputs changes relative to other.
        nout = other.nout + max(self.nout - other.nin, 0)
        other_op_maps = other._adjusted_maps(
            0, nin - other.nin, nin - other.nin + nout - other.nout)
        # finally change where other gets its inputs.
        ou0_nin = other.ufuncs[0].nin
        other_op_maps[0] = ([other_input_remap[i]
                             for i in other_op_maps[0][:ou0_nin]] +
                            other_op_maps[0][ou0_nin:])

        ufuncs = self.ufuncs + other.ufuncs
        op_maps = self_op_maps + other_op_maps
        ntmp = max(self.nout + self.ntmp - nout,
                   other.nout + other.ntmp - nout, 0)
        names = (self.names[:self.nin] +
                 other.names[self.nout:self.nout + extra_nin] +
                 [(other.names[i]
                   if other.names[i] else
                   self.names[self.nin + i])
                  for i in range(self.nout)] +
                 other.names[other.nin + n_other_in_from_self_out:other.nargs] +
                 self.names[self.nargs:] +
                 other.names[other.nargs:other.nargs + other.ntmp - self.ntmp])

        return cls(ChainedUfunc(ufuncs, op_maps, nin, nout, ntmp, names))

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
        return "WrappedUfunc({})".format(self.ufunc)

    def _arg_name(self, i):
        name = self.names[i]
        if name is None:
            if i < self.nin:
                name = 'in{}'.format(i)
            elif i < self.nin + self.nout:
                name = 'out{}'.format(i-self.nin)
            else:
                name = 'tmp{}'.format(i-self.nargs)
        return name

    def graph(self):
        from graphviz import Digraph

        dg_in = Digraph('in', node_attr=dict(shape='point', rank='min'))
        for i in range(self.nin):
            dg_in.node(self._arg_name(i))
        dg_out = Digraph('out', node_attr=dict(shape='point', rank='max'))
        for i in range(self.nout):
            dg_out.node(self._arg_name(self.nin + i))

        dg = Digraph(graph_attr=dict(rankdir='LR'))
        dg.subgraph(dg_in)
        dg.subgraph(dg_out)
        array_label = ' | '.join(
            ['<{name}> {name}'.format(name=self._arg_name(i))
             for i in range(self.nargs+self.ntmp)])
        dg_arrays = Digraph('arrays', node_attr=dict(
            shape='record', group='arrays', label=array_label))
        for iu in range(len(self.ufuncs) + 1):
            dg_arrays.node('node{}'.format(iu))
        dg.subgraph(dg_arrays)

        # Link inputs to node0.
        node_port = "node{}:{}".format
        for i in range(self.nin):
            arg_name = self._arg_name(i)
            dg.edge(arg_name, node_port(0, arg_name))
        for iu, (ufunc, op_map) in enumerate(
                zip(self.ufuncs, self.op_maps)):

            # ensure array holders are aligned
            arg_name = self._arg_name(0)
            dg.edge(node_port(iu, arg_name), node_port(iu+1, arg_name),
                    style='invis')
            # connect arrays to ufunc inputs.
            name = ufunc.__name__
            for i in range(ufunc.nin):
                arg_name = self._arg_name(op_map[i])
                if ufunc.nin == 1:
                    extra = dict()
                else:
                    extra = dict(headlabel=str(i))
                dg.edge(node_port(iu, arg_name), name, **extra)
            # connect ufunc outputs to next array.
            for i in range(ufunc.nout):
                arg_name = self._arg_name(op_map[ufunc.nin+i])
                if ufunc.nout == 1:
                    extra = dict()
                else:
                    extra = dict(taillabel=str(i))
                dg.edge(name, node_port(iu+1, arg_name), **extra)
        # finally, connect last array to outputs.
        for i in range(self.nout):
            arg_name = self._arg_name(self.nin+i)
            dg.edge(node_port(len(self.ufuncs), arg_name), arg_name)

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
                print('>1 output for non-Input input')
                return NotImplemented
            result |= ufunc
            if input_first:
                op_maps = result._adjusted_maps(1, 0, 0)
                op_maps[-1][ufunc.nin-1] = 0
                result.ufunc.op_maps[:] = op_maps
                result.names[:result.nin] = ([input_.name] +
                                             result.names[:result.nin-1])
            else:
                result.names[result.ufunc.nin - 1] = input_.name

        else:
            result = WrappedUfunc(ufunc)

            names = [a.name for a in inputs]
            if len(names) - names.count(None) != len(set(names) - {None}):
                print("duplicate names")

            # combine inputs
            result.ufunc.names[:ufunc.nin] = names

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

        # For now, only copy names
        for iout, output in zip(result.op_maps[-1][ufunc.nin:],
                                outputs):
            if output is not None:
                result.names[iout] = output.name
        return result
