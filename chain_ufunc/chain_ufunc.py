import numpy as np


__all__ = ['ChainedUfunc', 'WrappedUfunc', 'Input']


class ChainedUfunc(object):
    """A chain of ufuncs that are evaluated in turn.

    Parameters
    ----------
    ufuncs : list of ufuc
        Ufuncs to calculate, in order.
    input_maps : list of list
        For each ufunc, indices of where to get its inputs.  Up to
        nin, these point to actual inputs, then the outputs, and then
        any temporary arrays.
    output_maps: list of list
        For each ufunc, where to store the outputs.  Same numbering as
        for the input maps, but can be output or temporary only.
    nin, nout, ntmp : int
        total number of inputs, outputs and temporary arrays
    names : list of str, optional
        Input argument names ("in<i>" by default).
    """
    def __init__(self, ufuncs, input_maps, output_maps, nin, nout, ntmp,
                 names=None):
        self.ufuncs = ufuncs
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.nin = nin
        self.nout = nout
        self.ntmp = ntmp
        self.nargs = nin+nout
        # should have something for different types.
        if names is None:
            names = [None] * nin
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
        for ufunc, input_map, output_map in zip(self.ufuncs, self.input_maps,
                                                self.output_maps):
            ufunc_inout = [arrays[i] for i in input_map + output_map]
            # As we work in-place, result is not needed.
            ufunc(*ufunc_inout)

        outputs = arrays[self.nin:self.nin+self.nout]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @classmethod
    def from_ufunc(cls, ufunc):
        """Wrap a ufunc as a ChainedUfunc.

        Parameters
        ----------
        ufunc : ufunc-like
            If already a ChainedUfunc, the instance will be returned directly.
        """
        if isinstance(ufunc, cls):
            return ufunc

        if not isinstance(ufunc, np.ufunc):
            raise TypeError("ufunc should be an 'np.ufunc' instance.")

        input_map = list(range(ufunc.nin))
        output_map = list(range(ufunc.nin, ufunc.nargs))
        return cls([ufunc], [input_map], [output_map],
                   ufunc.nin, ufunc.nout, 0)

    def __repr__(self):
        return ("ChainedUfunc(ufuncs={ufuncs}, "
                "input_maps={input_maps}, "
                "output_maps={output_maps}, "
                "nin={nin}, "
                "nout={nout}, "
                "ntmp={ntmp}, "
                "names={names})").format(
                    ufuncs=self.ufuncs,
                    input_maps=self.input_maps,
                    output_maps=self.output_maps,
                    nin=self.nin,
                    nout=self.nout,
                    ntmp=self.ntmp,
                    names=self.names)

    def digraph(self):
        from graphviz import Digraph

        dg_in = Digraph('in', node_attr=dict(shape='point', rank='min'))
        for i in range(self.nin):
            dg_in.node('in{}'.format(i))
        dg_out = Digraph('out', node_attr=dict(shape='point', rank='max'))
        for i in range(self.nout):
            dg_out.node('out{}'.format(i))

        dg = Digraph(graph_attr=dict(rankdir='LR'))
        dg.subgraph(dg_in)
        dg.subgraph(dg_out)
        array_label = ' | '.join(
            ['<{t}{i}> {t}{i}'.format(t=t, i=i)
             for n, t in ((self.nin, 'in'),
                          (self.nout, 'out'),
                          (self.ntmp, 'tmp'))
             for i in range(n)])
        dg_arrays = Digraph('arrays', node_attr=dict(
            shape='record', group='arrays', label=array_label))
        for iu in range(len(self.ufuncs) + 1):
            dg_arrays.node('node{}'.format(iu))
        dg.subgraph(dg_arrays)

        def array(iu, i):
            if i < self.nin:
                inout = 'in'
            elif i < self.nin + self.nout:
                inout = 'out'
                i -= self.nin
            else:
                inout = 'tmp'
                i -= self.nin + self.nout
            return 'node{}:{}{}'.format(iu, inout, i)

        # Link inputs to node0.
        for i in range(self.nin):
            dg.edge('in{}'.format(i), array(0, i))
        for iu, (ufunc, input_map, output_map) in enumerate(
                zip(self.ufuncs, self.input_maps, self.output_maps)):

            # ensure array holders are aligned
            dg.edge(array(iu, 0), array(iu+1, 0), style='invis')
            # connect arrays to ufunc inputs.
            name = ufunc.__name__
            for i in range(ufunc.nin):
                if ufunc.nin == 1:
                    extra = dict()
                else:
                    extra = dict(headlabel=str(i))
                dg.edge(array(iu, input_map[i]), name, **extra)
            # connect ufunc outputs to next array.
            for i in range(ufunc.nout):
                if ufunc.nout == 1:
                    extra = dict()
                else:
                    extra = dict(taillabel=str(i))
                dg.edge(name, array(iu+1, output_map[i]), **extra)
        # finally, connect last array to outputs.
        for i in range(self.nout):
            dg.edge(array(len(self.ufuncs), self.nin + i),
                    'out{}'.format(i))

        return dg


class WrappedUfunc(object):
    """Wraps a ufunc so it can be used to construct chains.

    Parameters
    ----------
    ufunc : ufunc-like (`~numpy.ufunc` or `ChainedUfunc`)
        Ufunc to wrap
    """
    def __init__(self, ufunc, outsel=None):
        self.ufunc = ChainedUfunc.from_ufunc(ufunc)
        if outsel and ufunc.nout == 1:
            raise IndexError("scalar ufunc does not support indexing.")
        self.outsel = outsel
        for attr in ('ufuncs', 'nin', 'nout', 'ntmp',
                     'input_maps', 'output_maps', 'names'):
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

    def _adjusted_maps(self, off_in, off_out, off_tmp,
                       map_names=('input_maps', 'output_maps')):
        if off_in == 0 and off_out == 0 and (off_tmp == 0 or self.ntmp == 0):
            return {name: getattr(self, name)[:] for name in map_names}

        nin = self.nin
        ninplusout = nin + self.nout
        all_maps = {}
        for name in map_names:
            old_maps = getattr(self, name)
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
            all_maps[name] = new_maps
        return all_maps

    def __and__(self, other):
        if not isinstance(other, WrappedUfunc):
            return NotImplemented

        # first adjust the input and output maps for self
        self_maps = self._adjusted_maps(0, other.nin, other.nin + other.nout)
        other_maps = other._adjusted_maps(self.nin, self.nin + self.nout,
                                          self.nin + self.nout)

        return self.__class__(ChainedUfunc(
            self.ufuncs + other.ufuncs,
            self_maps['input_maps'] + other_maps['input_maps'],
            self_maps['output_maps'] + other_maps['output_maps'],
            self.nin + other.nin, self.nout + other.nout,
            max(self.ntmp, other.ntmp),
            self.names + other.names))

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
        for other_input_map in other.input_maps[1:]:
            if any(i < n_other_in_from_self_out for i in other_input_map):
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
        self_maps = self._adjusted_maps(0, extra_nin, extra_nin)

        # Now see how the number of outputs changes relative to other.
        nout = other.nout + max(self.nout - other.nin, 0)
        other_maps = other._adjusted_maps(0, nin - other.nin,
                                          nin - other.nin + nout - other.nout)
        # finally change where other gets its inputs.
        other_input_maps = ([[other_input_remap[i]
                              for i in other_maps['input_maps'][0]]] +
                            other_maps['input_maps'][1:])

        # Set up for in-place update
        ufuncs = self.ufuncs + other.ufuncs
        input_maps = self_maps['input_maps'] + other_input_maps
        output_maps = self_maps['output_maps'] + other_maps['output_maps']
        ntmp = max(self.nout + self.ntmp - nout,
                   other.nout + other.ntmp - nout, 0)
        names = self.names
        if extra_nin:
            names += other.names[self.nout:]

        return cls(ChainedUfunc(ufuncs, input_maps, output_maps,
                                nin, nout, ntmp, names))

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

    def digraph(self):
        return self.ufunc.digraph()


class Input(object):
    def __init__(self, name=None):
        self.name = name
        self.names = [name]

    def _can_handle(self, ufunc, method, *inputs, **kwargs):
        can_handle = ('out' not in kwargs and method == '__call__' and
                      all(isinstance(a, (Input, WrappedUfunc))
                          for a in inputs))
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # we're a mapping, and should turn the ufunc that called us
        # into a chainable version.
        if not self._can_handle(ufunc, method, *inputs, **kwargs):
            print(ufunc, method, inputs, kwargs)
            return NotImplemented

        if not all(isinstance(a, Input) for a in inputs):
            if ufunc.nin > 2:
                raise NotImplementedError('>2 inputs, with some not Input')

            self_first = self is inputs[0]
            result = inputs[self_first]
            if result.ufunc.nout > 1:
                print('>1 output for non-Input input')
                return NotImplemented
            result |= ufunc
            if self_first:
                input_maps = result._adjusted_maps(
                    1, 0, 0, map_names=('input_maps',))['input_maps']
                input_maps[-1][-1] = 0
                result.ufunc.input_maps = input_maps
                result.names[:] = [self.name] + result.names[:result.nin]
            else:
                result.names[result.ufunc.nin - 1] = self.name

            return result

        else:
            result = WrappedUfunc(ufunc)

            names = [a.name for a in inputs]
            if len(names) - names.count(None) != len(set(names) - {None}):
                print("duplicate names")

            # combine inputs
            result.ufunc.names = names
            return result
