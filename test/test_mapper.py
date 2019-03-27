import pytest
import numpy as np
np.random.seed(123)

import dyneusr as ds
from dyneusr.mapper.wrappers import KMapperWrapper
from dyneusr.mapper.utils import density_filter


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




class TestUtils:
    """ Simple tests for mapper.utils.py
    """

    def test_density_filter(self):
        mapper = KMapperWrapper(verbose=0)
        data_a = np.random.rand(100, 3) 
        data_b = np.random.rand(100, 3) * 10

        lens_a = density_filter(data_a, k=3)
        lens_b = density_filter(data_b, k=3)

        assert data_a.shape[0] == lens_a.shape[0]
        assert data_b.shape[0] == lens_b.shape[0]
        assert lens_a.shape[1] == 3
        assert lens_b.shape[1] == 3

        # make sure that b has lower lowest density
        assert lens_a.min() > lens_b.min()
        return 


