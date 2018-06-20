import numpy as np


__all__ = ['ChainedUfunc', 'Mapping', 'GetItem', 'Input']


class ChainedUfunc(object):
    """Calculates a chain of ufuncs.

    Parameters
    ----------
    ufuncs : list of ufuc
        Ufuncs to calculate, in order
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
        # should have something for different types.
        if names is None:
            names = [None] * nin
        self.names = names

    def __eq__(self, other):
        return (type(other) is type(self) and
                self.__dict__ == other.__dict__)

    def __call__(self, *args, **kwargs):
        """Evaluate the ufunc.

        All inputs should be in args, an output can be given in kwargs.
        """
        if len(args) != self.nin:
            raise ValueError("invalid number of arguments.")

        args = tuple(np.asanyarray(arg) for arg in args)
        outputs = kwargs.pop('out', None)
        if outputs is None:
            shape = np.broadcast(*args).shape
            dtype = np.common_type(*args)
            outputs = tuple(np.zeros(shape, dtype) for i in range(self.nout))
        elif not isinstance(outputs, tuple):
            outputs = (outputs,)

        # TODO: check for None in outputs.
        if len(outputs) != self.nout:
            raise ValueError("invalid number of outputs")

        if self.ntmp > 0:
            print("have temporaties", self.ufuncs)
            temporaries = tuple(np.zeros_like(outputs[0])
                                for i in range(self.ntmp))
        else:
            temporaries = ()

        arrays = list(args) + list(outputs) + list(temporaries)
        for ufunc, input_map, output_map in zip(self.ufuncs, self.input_maps,
                                                self.output_maps):
            ufunc_in = tuple(arrays[i] for i in input_map)
            ufunc_out = tuple(arrays[i] for i in output_map)
            ufunc_res = ufunc(*ufunc_in, out=ufunc_out)
            # mostly for mapping, put result back in arrays.
            if not isinstance(ufunc_res, tuple):
                ufunc_res = (ufunc_res,)
            for i, res in zip(output_map, ufunc_res):
                arrays[i] = res
            print(arrays)

        outputs = arrays[self.nin:self.nin+self.nout]

        return outputs[0] if len(outputs) == 1 else outputs

    def _can_handle(self, ufunc, method, *inputs, **kwargs):
        can_handle = ('out' not in kwargs and method == '__call__' and
                      all(isinstance(a, ChainedUfunc) for a in inputs))
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not self._can_handle(ufunc, method, *inputs, **kwargs):
            print(ufunc, method, inputs, kwargs)
            return NotImplemented

        # combine inputs
        combined_input = inputs[0]
        for input_ in inputs[1:]:
            combined_input &= input_

        return self.from_links([combined_input, ufunc])

    def __or__(self, other):
        return self.from_links([self, other])

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
        if not isinstance(other, ChainedUfunc):
            return NotImplemented

        # check if both are just different parts of the same ChainedUfunc
        if (isinstance(self.ufuncs[-1], GetItem) and
            isinstance(other.ufuncs[-1], GetItem) and
            (self.ufuncs[-1].nout + other.ufuncs[-1].nout ==
             self.ufuncs[-1].nin and
             self.ufuncs[-1].mapping + other.ufuncs[-1].mapping ==
             list(range(self.ufuncs[-1].nin))) and
            (self.ufuncs[:-1] == other.ufuncs[:-1] and
             self.input_maps[:-1] == other.input_maps[:-1] and
             self.output_maps[:-1] == other.output_maps[:-1])):
            # the two add up to a complete map, which can just be removed.
            return type(self)(self.ufuncs[:-1], self.input_maps[:-1],
                              self.output_maps[:-1], self.nin,
                              self.ufuncs[-1].nin, self.ntmp)

        # first adjust the input and output maps for self
        self_maps = self._adjusted_maps(0, other.nin, other.nin + other.nout)
        other_maps = other._adjusted_maps(self.nin, self.nin + self.nout,
                                          self.nin + self.nout)

        return type(self)(self.ufuncs + other.ufuncs,
                          self_maps['input_maps'] + other_maps['input_maps'],
                          self_maps['output_maps'] + other_maps['output_maps'],
                          self.nin + other.nin, self.nout + other.nout,
                          max(self.ntmp, other.ntmp),
                          self.names + other.names)

    @classmethod
    def from_ufunc(cls, ufunc):
        if not isinstance(ufunc, np.ufunc):
            raise TypeError("ufunc should be an 'np.ufunc' instance.")
        input_map = list(range(ufunc.nin))
        output_map = list(range(ufunc.nin, ufunc.nargs))
        return cls([ufunc], [input_map], [output_map],
                   ufunc.nin, ufunc.nout, 0)

    def copy(self):
        return self.from_link(self)

    @classmethod
    def from_link(cls, link):
        if isinstance(link, np.ufunc):
            return cls.from_ufunc(link)

        return cls(link.ufuncs, link.input_maps, link.output_maps,
                   link.nin, link.nout, link.ntmp, link.names)

    @classmethod
    def from_links(cls, links):
        result = cls.from_link(links[0])
        for link in links[1:]:
            result.append(link)
        return result

    def append(self, other):
        """In-place addition of a single link."""
        if not isinstance(other, ChainedUfunc):
            if isinstance(other, np.ufunc):
                other = ChainedUfunc.from_ufunc(other)
            else:
                raise TypeError("link should be a (chained) ufunc.")

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

        # Update in-place.
        self.__init__(ufuncs, input_maps, output_maps, nin, nout, ntmp, names)

    def __getitem__(self, item):
        result = self.copy()
        result.append(GetItem(item, self.nout))
        return result

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


class Mapping(ChainedUfunc):
    """Map inputs to outputs.

    Parameters
    ----------
    mapping : list of int
        Remapped outputs
    nin : number of inputs, optional
        By default, equal to the number of mapped items.
    """
    def __init__(self, mapping=[0], nin=None, names=None):
        nout = len(mapping)
        if nin is None:
            nin = nout
        self.mapping = mapping
        super(Mapping, self).__init__([self], [list(range(nin))],
                                      [list(range(nin, nin+nout))],
                                      nin, nout, 0, names=names)

    def copy(self):
        return self.__class__(self.mapping, self.nin)

    def __call__(self, *inputs, **kwargs):
        for input_ in inputs:
            if isinstance(input_, ChainedUfunc):
                result = input_.__array_ufunc__(self, '__call__',
                                                *inputs, **kwargs)
                if result is NotImplemented:
                    raise TypeError("mapping not implemented for these types.")
                else:
                    return result

        outputs = tuple(inputs[r] for r in self.mapping)
        return outputs[0] if len(outputs) == 1 else outputs

    def __or__(self, other):
        if isinstance(other, np.ufunc):
            other = ChainedUfunc.from_ufunc(other)
        elif isinstance(other, ChainedUfunc):
            other = other.copy()
        else:
            return NotImplemented

        if self.nout != other.nin:
            raise TypeError('cannot or with chain with non-matching nin')

        other.input_maps[0] = self.input_maps[0]
        return other

    def __and__(self, other):
        if not isinstance(other, Mapping) and type(other) is ChainedUfunc:
            return super(Mapping, self).__and__(other)

        return type(self)(self.mapping +
                          [i + self.nin for i in other.mapping],
                          names=self.names + other.names)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # we're a mapping, and should turn the ufunc that called us
        # into a chainable version.
        if not self._can_handle(ufunc, method, *inputs, **kwargs):
            print(ufunc, method, inputs, kwargs)
            return NotImplemented

        if not all(isinstance(a, Mapping) for a in inputs):
            print('not all Mapping')
            return NotImplemented

        # combine inputs
        combined_input = inputs[0]
        for input_ in inputs[1:]:
            combined_input &= input_

        result = ChainedUfunc.from_ufunc(ufunc)
        result.input_maps[0] = combined_input.mapping
        return result

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.mapping == other.mapping)

    def append(self, other):
        if not isinstance(other, GetItem):
            raise TypeError("can only append GetItem.")
        if self.mapping != list(range(self.nin)):
            raise NotImplementedError("only support inputs")

        new_map = self.mapping[other.item]
        if not isinstance(new_map, list):
            new_map = [new_map]

        self.mapping = new_map
        self.nout = len(new_map)

    def __repr__(self):
        return ("Mapping({0}, {1}, names={2})"
                .format(self.mapping, self.nin, self.names))


class GetItem(Mapping):
    def __init__(self, item, nin):
        self.item = item
        input_map = list(range(nin))[item]
        if not isinstance(input_map, list):
            input_map = [input_map]
        super(GetItem, self).__init__(input_map, nin)

    def copy(self):
        return self.__class__(self.item, self.nin)


class Input:
    nin = 1
    nout = 1
    nargs = 2

    def __init__(self, name=None):
        self.name = name
        self.names = [name]

    def _can_handle(self, ufunc, method, *inputs, **kwargs):
        can_handle = ('out' not in kwargs and method == '__call__' and
                      all(isinstance(a, (Input, ChainedUfunc))
                          for a in inputs))
        return can_handle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # we're a mapping, and should turn the ufunc that called us
        # into a chainable version.
        if not self._can_handle(ufunc, method, *inputs, **kwargs):
            print(ufunc, method, inputs, kwargs)
            return NotImplemented

        result = ChainedUfunc.from_ufunc(ufunc)

        if not all(isinstance(a, Input) for a in inputs):
            if ufunc.nin > 2:
                print('>2 inputs, with some not Input')
                return NotImplemented
            if any(a.nout > 1 for a in inputs):
                print('>1 output for some input')
                return NotImplemented
            # self_first = self is inputs[0]
            # other = inputs[1] if self_first else inputs[0]
            # new_maps = other._adjusted_maps(self_first, other1, 1)
            # new_maps = inputs[1]._adjusted_maps(1, 1, 1)

        names = [a.name for a in inputs]
        if len(names) - names.count(None) != len(set(names) - {None}):
            print("duplicate names")
        # combine inputs
        combined_input = Mapping(list(range(ufunc.nin)), names=names)

        result.input_maps[0] = combined_input.mapping
        result.names = combined_input.names
        return result
