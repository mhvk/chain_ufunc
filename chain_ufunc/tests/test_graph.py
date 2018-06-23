import pytest
import numpy as np
from ..chain_ufunc import Input

pytest.importorskip('graphviz')


class TestGraph:
    def test_graph_creation(self, tmpdir):
        muladd = np.add(Input(), np.multiply(Input(), Input()))
        graph = muladd.graph()
        graph.render(str(tmpdir.join('graph.gv')))
