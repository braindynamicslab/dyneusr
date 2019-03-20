import pytest
import numpy as np

import dyneusr as ds
from dyneusr.core import DyNeuGraph
from dyneusr.mapper.wrappers import KMapperWrapper


class TestDyNeuGraph:
    """ Simple tests for DyNeuGraph
    """
    def test_init(self):
        mapper = KMapperWrapper(verbose=0)
        data = np.random.rand(100, 3)
        graph = mapper.fit_map(data)

        dG = DyNeuGraph(G=graph)

        assert dG.A.shape[0] == len(graph['nodes'])
        assert dG.TCM.shape[0] == data.shape[0]
        return 


    def test_init_with_y(self):
        mapper = KMapperWrapper(verbose=0)
        data = np.random.rand(100, 3)
        graph = mapper.fit_map(data)
        y = mapper.lens_

        dG = DyNeuGraph(G=graph, y=y)

        assert dG.A.shape[0] == len(graph['nodes'])
        assert dG.TCM.shape[0] == data.shape[0]
        return 
