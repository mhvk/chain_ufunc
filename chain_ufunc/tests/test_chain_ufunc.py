import numpy as np
from ..chain_ufunc import Mapping


def test_or():
    # simple tests.
    one_input = Mapping()
    two_inputs = Mapping([0, 1])
    mulsin = two_inputs | np.multiply | np.sin
    degrees = np.array([0., 30., 90., 150., 180., 210., 270., 330., 360.])
    deg2rad = np.array(np.pi/180.)
    tst = mulsin(degrees, deg2rad)
    assert np.all(tst == np.sin(degrees * deg2rad))
    mulsinarcsin = two_inputs | np.multiply | np.sin | np.arcsin
    tst = mulsinarcsin(degrees, deg2rad)
    assert np.allclose(tst, np.arcsin(np.sin(degrees * deg2rad)))
    mulsinarcsin2 = two_inputs | mulsin | np.arcsin
    assert mulsinarcsin2 == mulsinarcsin
    addmodf = two_inputs | np.add | np.modf
    in1 = np.array([1.5, 2.])
    in2 = np.array([0.1, -0.1])
    tst = addmodf(in1, in2)
    chck = np.modf(np.add(in1, in2))
    assert np.all(tst[0] == chck[0]) and np.all(tst[1] == chck[1])
    modfneg = one_input | np.modf | np.negative
    tst = modfneg(in1)
    assert (np.all(tst[0] == -np.modf(in1)[0]) and
            np.all(tst[1] == np.modf(in1)[1]))
    mapping = Mapping([1, 0])
    modfmap = one_input | np.modf | mapping
    tst = modfmap(in1)
    assert (np.all(tst[0] == np.modf(in1)[1]) and
            np.all(tst[1] == np.modf(in1)[0]))
    modfmapneg = one_input | np.modf | mapping | np.negative
    tst = modfmapneg(in1)
    assert (np.all(tst[0] == -np.modf(in1)[1]) and
            np.all(tst[1] == np.modf(in1)[0]))
    modfmapnegmap = one_input | np.modf | mapping | np.negative | mapping
    tst = modfmapnegmap(in1)
    assert (np.all(tst[0] == np.modf(in1)[0]) and
            np.all(tst[1] == -np.modf(in1)[1]))
    muladd = two_inputs | np.multiply | np.add
    tst = muladd(in1, np.array(3.), in2)
    assert np.all(tst == in1 * 3. + in2)
    modfadd = one_input | np.modf | np.add
    tst = modfadd(in1)
    assert np.all(tst == in1)
    modfmul = one_input | np.modf | np.multiply
    tst = modfmul(in1)
    assert np.all(tst == np.modf(in1)[0] * np.modf(in1)[1])
    modfaddmodf = one_input | np.modf | np.add | np.modf
    tst = modfaddmodf(in1)
    assert (np.all(tst[0] == np.modf(in1)[0]) and
            np.all(tst[1] == np.modf(in1)[1]))
    negmapping = one_input | np.negative | mapping
    tst = negmapping(in1, in2)
    assert np.all(tst[0] == in2) and np.all(tst[1] == -in1)


def test_and():
    one_input = Mapping()
    two_inputs = Mapping([0, 1])
    two_inputs2 = one_input & one_input
    assert two_inputs == two_inputs2
    mul = two_inputs | np.multiply
    mul2 = mul & mul
    in1 = np.array([1.5, 2.])
    in2 = np.array([0.1, -0.1])
    tst = mul2(in1, 2., in2, in1)
    assert np.all(tst == np.vstack((in1*2, in2*in1)))
    neg = one_input | np.negative
    return
    # following does not work yet, since we cannot handle parallel chains.
    modfneg2 = one_input | np.modf | (neg & neg)
    tst = modfneg2(in1)
    assert (np.all(tst[0] == -np.modf(in1)[0]) and
            np.all(tst[1] == -np.modf(in1)[1]))


def test_array_ufunc():
    # simple tests.
    input = Mapping()
    mulsin = np.sin(np.multiply(input, input))
    degrees = np.array([0., 30., 90., 150., 180., 210., 270., 330., 360.])
    deg2rad = np.array(np.pi/180.)
    tst = mulsin(degrees, deg2rad)
    assert np.all(tst == np.sin(degrees * deg2rad))
    mulsinarcsin = np.arcsin(np.sin(np.multiply(input, input)))
    tst = mulsinarcsin(degrees, deg2rad)
    assert np.allclose(tst, np.arcsin(np.sin(degrees * deg2rad)))
    mulsinarcsin2 = np.arcsin(mulsin)
    assert mulsinarcsin2 == mulsinarcsin
    addmodf = np.modf(np.add(input, input))
    in1 = np.array([1.5, 2.])
    in2 = np.array([0.1, -0.1])
    tst = addmodf(in1, in2)
    chck = np.modf(np.add(in1, in2))
    assert np.all(tst[0] == chck[0]) and np.all(tst[1] == chck[1])
    two_inputs = Mapping([0, 1])
    muladd = np.add(np.multiply(*two_inputs), input)
    # muladd = np.add(np.multiply(input, input), input)
    tst = muladd(in1, np.array(3.), in2)
    assert np.all(tst == in1 * 3. + in2)
    chck = np.modf(input)[1]
    tst = chck(in1)
    assert np.all(tst == np.modf(in1)[1])
    modfadd = np.add(*np.modf(input))
    tst = modfadd(in1)
    assert np.all(tst == in1)
    modfmul = np.multiply(*np.modf(input))
    tst = modfmul(in1)
    assert np.all(tst == np.modf(in1)[0] * np.modf(in1)[1])
    modfaddmodf = np.modf(np.add(*np.modf(input)))
    tst = modfaddmodf(in1)
    assert (np.all(tst[0] == np.modf(in1)[0]) and
            np.all(tst[1] == np.modf(in1)[1]))
    mapping = Mapping([1, 0])
    modfmap = mapping(*np.modf(input))
    tst = modfmap(in1)
    assert (np.all(tst[0] == np.modf(in1)[1]) and
            np.all(tst[1] == np.modf(in1)[0]))
