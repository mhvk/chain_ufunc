import pytest
import numpy as np
from ..chain_ufunc import Input

pytest.importorskip('graphviz')


class TestDigraph:
    def test_digraph_creation(self, tmpdir):
        muladd = np.add(Input(), np.multiply(Input(), Input()))
        digraph = muladd.digraph()
        digraph.render(str(tmpdir.join('digraph.gv')))
