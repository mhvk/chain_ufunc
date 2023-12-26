import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from chain_ufunc import create_chained_ufunc, create_from_doc, get_chain


class TestGetChain:
    def test_regular_ufunc(self):
        chain = get_chain(np.add)
        assert chain == [(np.add, [0, 1, 2])]
        chain = get_chain(np.modf)
        assert chain == [(np.modf, [0, 1, 2])]
        chain = get_chain(np.sin)
        assert chain == [(np.sin, [0, 1])]

    def test_chain(self):
        links = [(np.sin, [0, 2]),
                 (np.cos, [1, 3]),
                 (np.arctan2, [2, 3, 2])]
        sincosarctan2 = create_chained_ufunc(links, 2, 1, 1)
        chain = get_chain(sincosarctan2)
        assert chain == links


class TestSimple:
    @classmethod
    def setup_class(self):
        self.degrees = np.array([0., 30., 90., 150., 180.,
                                 210., 270., 330., 360.])
        self.deg2rad = np.array(np.pi/180.)

    def test_one_function(self):
        mul = create_chained_ufunc([(np.multiply, [0, 1, 2])], 2, 1, 0)
        tst = mul(self.degrees, self.deg2rad)
        assert_array_equal(tst, self.degrees * self.deg2rad)

    def test_two_functions(self):
        mulsin = create_chained_ufunc([(np.multiply, [0, 1, 2]),
                                       (np.sin, [2, 2])], 2, 1, 0)
        tst = mulsin(self.degrees, self.deg2rad)
        assert_array_equal(tst, np.sin(self.degrees * self.deg2rad))

    def test_function_of_two_functions(self):
        links = [(np.sin, [0, 2]),
                 (np.cos, [1, 3]),
                 (np.arctan2, [2, 3, 2])]
        sincosarctan2 = create_chained_ufunc(links, 2, 1, 1)
        assert get_chain(sincosarctan2) == links
        angles = self.degrees * self.deg2rad
        tst = sincosarctan2(angles, angles)
        chck = np.arctan2(np.sin(angles), np.cos(angles))
        assert_array_almost_equal(tst, chck)

    def test_two_outputs(self):
        addmodf = create_chained_ufunc([(np.add, [0, 1, 2]),
                                        (np.modf, [2, 2, 3])], 2, 2, 0)
        in1 = np.array([1.5, 2.])
        in2 = np.array([0.1, -0.1])
        tst = addmodf(in1, in2)
        chck = np.modf(np.add(in1, in2))
        assert_array_equal(tst[0], chck[0])
        assert_array_equal(tst[1], chck[1])


class TestCache:
    uf = staticmethod(
        create_chained_ufunc([(np.multiply, [0, 1, 4]),
                              (np.multiply, [2, 3, 5]),
                              (np.add, [4, 5, 4])],
                             4, 1, 1))

    @staticmethod
    def fun(a, b, c, d):
        return a*b + c*d

    @pytest.mark.parametrize("array", [
        np.arange(10000.),
        np.broadcast_to(1., (10000,))
    ])
    def test_with_array_larger_than_bufsize(self, array):
        result = self.uf(array, 2., 3., 4.)
        expected = self.fun(array, 2., 3., 4.)
        assert_array_equal(result, expected)
        result = self.uf(1., 2., 3., array)
        expected = self.fun(1., 2., 3., array)
        assert_array_equal(result, expected)


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
        assert_array_equal(tst, (in1 * in2) + 3.)

        def chain(a, b, c):
            d = None
            d = multiply(a, b, out=d)
            d = add(d, c, out=d)
            return d

        assert_array_equal(chain(in1, in2, 3.), tst)
