import numpy as np
from ..chain_ufunc import Input, Output


class TestSimple:
    def test_one_function(self):
        mul = np.multiply(Input(), Input())
        assert mul.names == [None, None, None]
        mul = np.multiply(Input('a'), Input('b'))
        assert mul.names == ['a', 'b', None]
        mul = np.multiply(Input('a'), Input())
        assert mul.names == ['a', None, None]
        mul = np.multiply(Input(), Input('b'))
        assert mul.names == [None, 'b', None]

    def test_one_function_out(self):
        mul = np.multiply(Input(), Input(), Output())
        assert mul.names == [None, None, None]
        mul = np.multiply(Input('a'), Input('b'), Output('c'))
        assert mul.names == ['a', 'b', 'c']
        mul = np.multiply(Input(), Input(), Output('c'))
        assert mul.names == [None, None, 'c']
        mul = np.multiply(Input(), Input('b'), Output('c'))
        assert mul.names == [None, 'b', 'c']

    def test_one_function_out_kwd(self):
        mul = np.multiply(Input(), Input(), out=Output())
        assert mul.names == [None, None, None]
        mul = np.multiply(Input('a'), Input('b'), out=Output('c'))
        assert mul.names == ['a', 'b', 'c']
        mul = np.multiply(Input(), Input(), out=Output('c'))
        assert mul.names == [None, None, 'c']
        mul = np.multiply(Input(), Input('b'), out=(Output('c'),))
        assert mul.names == [None, 'b', 'c']

    def test_two_functions(self):
        mulsin = np.sin(np.multiply(Input(), Input()))
        assert mulsin.names == [None, None, None]
        mulsin = np.sin(np.multiply(Input(), Input(), Output()))
        assert mulsin.names == [None, None, None]
        mulsin = np.sin(np.multiply(Input(), Input()), Output())
        assert mulsin.names == [None, None, None]
        mulsin = np.sin(np.multiply(Input(), Input(), Output()), Output())
        assert mulsin.names == [None, None, None]
        mulsin = np.sin(np.multiply(Input('a'), Input(), Output('c')))
        assert mulsin.names == ['a', None, 'c']
        mulsin = np.sin(np.multiply(Input('a'), Input()), Output('d'))
        assert mulsin.names == ['a', None, 'd']
        mulsin = np.sin(np.multiply(Input('a'), Input(), Output('c')),
                        Output('d'))
        assert mulsin.names == ['a', None, 'd']
        mulsin = np.sin(np.multiply(Input('a'), Input('b')), Output('d'))
        assert mulsin.names == ['a', 'b', 'd']

    def test_function_of_two_functions(self):
        sincosarctan2 = np.arctan2(np.sin(Input()), np.cos(Input()))
        assert sincosarctan2.names == [None, None, None, None]
        sincosarctan2 = np.arctan2(np.sin(Input('sa')), np.cos(Input()))
        assert sincosarctan2.names == ['sa', None, None, None]
        sincosarctan2 = np.arctan2(np.sin(Input()), np.cos(Input('ca')))
        assert sincosarctan2.names == [None, 'ca', None, None]
        sincosarctan2 = np.arctan2(np.sin(Input('sa')), np.cos(Input('ca')))
        assert sincosarctan2.names == ['sa', 'ca', None, None]
        sincosarctan2 = np.arctan2(np.sin(Input()), np.cos(Input()),
                                   Output('r'))
        assert sincosarctan2.names == [None, None, 'r', None]
        sincosarctan2 = np.arctan2(np.sin(Input('sa'), Output('s')),
                                   np.cos(Input('ca')))
        assert sincosarctan2.names == ['sa', 'ca', 's', None]
        sincosarctan2 = np.arctan2(np.sin(Input('sa')),
                                   np.cos(Input('ca'), Output('c')))
        assert sincosarctan2.names == ['sa', 'ca', None, 'c']
        sincosarctan2 = np.arctan2(np.sin(Input('sa'), Output('s')),
                                   np.cos(Input('ca'), Output('c')))
        assert sincosarctan2.names == ['sa', 'ca', 's', 'c']
        sincosarctan2 = np.arctan2(np.sin(Input('sa'), Output('s')),
                                   np.cos(Input('ca'), Output('c')),
                                   Output('r'))
        assert sincosarctan2.names == ['sa', 'ca', 'r', 'c']

    def test_two_outputs(self):
        addmodf = np.modf(np.add(Input(), Input()))
        assert addmodf.names == [None, None, None, None]
        addmodf = np.modf(np.add(Input(), Input(), Output()))
        assert addmodf.names == [None, None, None, None]
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          Output())
        assert addmodf.names == [None, None, None, None]
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          Output(), Output())
        assert addmodf.names == [None, None, None, None]
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          None, Output())
        assert addmodf.names == [None, None, None, None]
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          Output('f'))
        assert addmodf.names == [None, None, 'f', None]
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          Output('f'), Output('i'))
        assert addmodf.names == [None, None, 'f', 'i']
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          None, Output('i'))
        assert addmodf.names == [None, None, None, 'i']
        addmodf = np.modf(np.add(Input(), Input(), Output()),
                          Output(), Output('i'))
        assert addmodf.names == [None, None, None, 'i']
        addmodf = np.modf(np.add(Input(), Input(), Output('sum')),
                          Output(), Output('i'))
        assert addmodf.names == [None, None, None, 'i']
        addmodf = np.modf(np.add(Input(), Input(), Output('sum')),
                          None, Output('i'))
        assert addmodf.names == [None, None, 'sum', 'i']
