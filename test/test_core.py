import pytest
import os
import tempfile
import numpy as np
np.random.seed(123)

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


    def test_visualize(self):
        mapper = KMapperWrapper(verbose=0)
        data = np.random.rand(100, 3)
        graph = mapper.fit_map(data)
        y = mapper.lens_

        dG = DyNeuGraph(G=graph, y=y)

        # visualize results in a tempdir
        with tempfile.TemporaryDirectory() as temp_dir:
	        temp_html = os.path.join(temp_dir, 'test.html')
	        dG.visualize(temp_html, path_assets=temp_dir, show=False, port=None)
	        assert os.path.exists(temp_html)
        return 


    def test_visualize_show(self):
        mapper = KMapperWrapper(verbose=0)
        data = np.random.rand(100, 3)
        graph = mapper.fit_map(data)
        y = mapper.lens_

        dG = DyNeuGraph(G=graph, y=y)

        # visualize results in a tempdir
        with tempfile.TemporaryDirectory() as temp_dir:
	        temp_html = os.path.join(temp_dir, 'test.html')
	        dG.visualize(temp_html, path_assets=temp_dir, show=False, port=None)
	        assert os.path.exists(temp_html)
        return 
