import pytest
import numpy as np

import dyneusr as ds
from dyneusr.mapper.wrappers import KMapperWrapper


class TestWrappers:
    """ Simple tests for mapper.wrappers.py
    """

    def test_KMapperWrapper(self):
        mapper = KMapperWrapper(verbose=0)
        data = np.random.rand(100, 3)
        graph = mapper.fit_map(data)
        assert mapper.lens_.shape[0] == data.shape[0]
        assert graph.get('nodes') and graph.get('links')
        return 


