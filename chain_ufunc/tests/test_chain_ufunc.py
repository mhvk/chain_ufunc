import numpy as np
from ..chain_ufunc import ChainedUfunc


class TestSimple:
    def setup(self):
        self.degrees = np.array([0., 30., 90., 150., 180.,
                                 210., 270., 330., 360.])
        self.deg2rad = np.array(np.pi/180.)

    def test_one_function(self):
        mul = ChainedUfunc([np.multiply], [[0, 1, 2]], 2, 1, 0)
        tst = mul(self.degrees, self.deg2rad)
        assert np.all(tst == self.degrees * self.deg2rad)

    def test_two_functions(self):
        mulsin = ChainedUfunc([np.multiply, np.sin],
                              [[0, 1, 2], [2, 2]], 2, 1, 0)
        tst = mulsin(self.degrees, self.deg2rad)
        assert np.all(tst == np.sin(self.degrees * self.deg2rad))

    def test_function_of_two_functions(self):
        sincosarctan2 = ChainedUfunc([np.sin, np.cos, np.arctan2],
                                     [[0, 2], [1, 3], [2, 3, 2]],
                                     2, 1, 1)
        angles = self.degrees * self.deg2rad
        tst = sincosarctan2(angles, angles)
        chck = np.arctan2(np.sin(angles), np.cos(angles))
        assert np.allclose(tst, chck)

    def test_two_outputs(self):
        addmodf = ChainedUfunc([np.add, np.modf],
                               [[0, 1, 2], [2, 2, 3]], 2, 2, 0)
        in1 = np.array([1.5, 2.])
        in2 = np.array([0.1, -0.1])
        tst = addmodf(in1, in2)
        chck = np.modf(np.add(in1, in2))
        assert np.all(tst[0] == chck[0]) and np.all(tst[1] == chck[1])
