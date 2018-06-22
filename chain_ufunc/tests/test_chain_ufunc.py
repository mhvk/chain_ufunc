import numpy as np
from ..chain_ufunc import Input


class TestSimple:
    def setup(self):
        self.degrees = np.array([0., 30., 90., 150., 180.,
                                 210., 270., 330., 360.])
        self.deg2rad = np.array(np.pi/180.)

    def test_two_functions(self):
        mulsin = np.sin(np.multiply(Input(), Input()))
        tst = mulsin(self.degrees, self.deg2rad)
        assert np.all(tst == np.sin(self.degrees * self.deg2rad))

    def test_three_functions(self):
        mulsinarcsin = np.arcsin(np.sin(np.multiply(Input(), Input())))
        tst = mulsinarcsin(self.degrees, self.deg2rad)
        assert np.allclose(tst, np.arcsin(np.sin(self.degrees *
                                                 self.deg2rad)))

    def test_function_of_two_functions(self):
        sincosarctan2 = np.arctan2(np.sin(Input()), np.cos(Input()))
        angles = self.degrees * self.deg2rad
        tst = sincosarctan2(angles, angles)
        chck = np.arctan2(np.sin(angles), np.cos(angles))
        assert np.allclose(tst, chck)

    def test_two_outputs(self):
        addmodf = np.modf(np.add(Input(), Input()))
        in1 = np.array([1.5, 2.])
        in2 = np.array([0.1, -0.1])
        tst = addmodf(in1, in2)
        chck = np.modf(np.add(in1, in2))
        assert np.all(tst[0] == chck[0]) and np.all(tst[1] == chck[1])

    def test_two_functions_three_inputs21(self):
        muladd = np.add(np.multiply(Input(), Input()), Input())
        tst = muladd(self.degrees, self.deg2rad, np.pi)
        assert np.all(tst == self.degrees * self.deg2rad + np.pi)

    def test_two_functions_three_inputs12(self):
        muladd = np.add(Input(), np.multiply(Input(), Input()))
        tst = muladd(np.pi, self.degrees, self.deg2rad)
        assert np.all(tst == np.pi + self.degrees * self.deg2rad)


class TestIndexing:
    def setup(self):
        self.in1 = np.array([1.5, 2.])

    def test_two_output_indexing(self):
        chck = np.modf(Input())[1]
        tst = chck(self.in1)
        assert np.all(tst == np.modf(self.in1)[1])

    def test_expansion(self):
        modfadd = np.add(*np.modf(Input()))
        tst = modfadd(self.in1)
        assert np.all(tst == self.in1)
        modfmul = np.multiply(*np.modf(Input()))
        tst = modfmul(self.in1)
        assert np.all(tst == np.modf(self.in1)[0] * np.modf(self.in1)[1])
        modfaddmodf = np.modf(np.add(*np.modf(Input())))
        tst = modfaddmodf(self.in1)
        assert (np.all(tst[0] == np.modf(self.in1)[0]) and
                np.all(tst[1] == np.modf(self.in1)[1]))


class TestIdentities:
    def test_chain_order_independence(self):
        mulsin = np.sin(np.multiply(Input(), Input()))
        mulsinarcsin = np.arcsin(np.sin(np.multiply(Input(), Input())))
        mulsinarcsin2 = np.arcsin(mulsin)
        assert mulsinarcsin == mulsinarcsin2
