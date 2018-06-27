import numpy as np
from .. import create_chained_ufunc, create_from_doc, get_chain


class TestGetChain:
    def test_regular_ufunc(self):
        chain = get_chain(np.add)
        assert chain == [(np.add, [0, 1, 2])]
        chain = get_chain(np.modf)
        assert chain == [(np.modf, [0, 1, 2])]
        chain = get_chain(np.sin)
        assert chain == [(np.sin, [0, 1])]

    def test_chain(self):
        sincosarctan2 = create_chained_ufunc([np.sin, np.cos, np.arctan2],
                                             [[0, 2], [1, 3], [2, 3, 2]],
                                             2, 1, 1)
        chain = get_chain(sincosarctan2)
        assert chain == [(np.sin, [0, 2]),
                         (np.cos, [1, 3]),
                         (np.arctan2, [2, 3, 2])]


class TestSimple:
    def setup(self):
        self.degrees = np.array([0., 30., 90., 150., 180.,
                                 210., 270., 330., 360.])
        self.deg2rad = np.array(np.pi/180.)

    def test_one_function(self):
        mul = create_chained_ufunc([np.multiply], [[0, 1, 2]], 2, 1, 0)
        tst = mul(self.degrees, self.deg2rad)
        assert np.all(tst == self.degrees * self.deg2rad)

    def test_two_functions(self):
        mulsin = create_chained_ufunc([np.multiply, np.sin],
                                      [[0, 1, 2], [2, 2]], 2, 1, 0)
        tst = mulsin(self.degrees, self.deg2rad)
        assert np.all(tst == np.sin(self.degrees * self.deg2rad))

    def test_function_of_two_functions(self):
        sincosarctan2 = create_chained_ufunc([np.sin, np.cos, np.arctan2],
                                             [[0, 2], [1, 3], [2, 3, 2]],
                                             2, 1, 1)
        angles = self.degrees * self.deg2rad
        tst = sincosarctan2(angles, angles)
        chck = np.arctan2(np.sin(angles), np.cos(angles))
        assert np.allclose(tst, chck)

    def test_two_outputs(self):
        addmodf = create_chained_ufunc([np.add, np.modf],
                                       [[0, 1, 2], [2, 2, 3]], 2, 2, 0)
        in1 = np.array([1.5, 2.])
        in2 = np.array([0.1, -0.1])
        tst = addmodf(in1, in2)
        chck = np.modf(np.add(in1, in2))
        assert np.all(tst[0] == chck[0]) and np.all(tst[1] == chck[1])


class TestCreateFromDoc:
    """
    def chain(a, b, c):
        d = None
        d = multiply(a, b, out=d)
        d = add(d, c, out=d)
        return d
    """
    def test_cls_doc(self):
        from numpy import multiply, add
        muladd = create_from_doc(self.__class__.__doc__)
        in1 = np.array([1.5, 2.])
        in2 = np.array([0.1, -0.1])
        tst = muladd(in1, in2, 3.)
        assert np.all(tst == (in1 * in2) + 3.)

        def chain(a, b, c):
            d = None
            d = multiply(a, b, out=d)
            d = add(d, c, out=d)
            return d

        assert np.all(chain(in1, in2, 3.) == tst)
