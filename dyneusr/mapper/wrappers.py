""" 
Flexible, sklearn-style wrappers for External Mapper 
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS 
from sklearn.cluster import DBSCAN

###############################################################################
### Optional imports
###############################################################################
# External mapper tools
try:
    from kmapper import KeplerMapper
    from kmapper.cover import Cover
except ImportError as e:
    print("[warning]", e)


# everything else should be in utils
from dyneusr.mapper.utils import optimize_cover, optimize_dbscan 



###############################################################################
### Helper Functions
###############################################################################
def _transform_lens(data=None, verbose=1, **params):
    """ Transform data into a lens using KeplerMapper. """
    mapper = KeplerMapper(verbose=verbose)
    return mapper.fit_transform(data, **params)


def _map_graph(lens=None, data=None, verbose=1, **params):
    """ Map lens, data into graph using KeplerMapper. """
    mapper = KeplerMapper(verbose=verbose)
    return mapper.map(lens, data, **params)

 

###############################################################################
### Base MapperWrapper
###############################################################################
class BaseMapperWrapper(BaseEstimator, TransformerMixin, ClusterMixin):

    def fit(self, data, y=None):
        pass

    def fit_transform(self, data, y=None):
        """ Transform data into lens 
        """
        return self.fit(data).lens_
    
    def fit_map(self, data, y=None):
        """ Fit lens, map data into graph.
        """
        return self.fit(data).graph_


###############################################################################
### External MapperWrappers
###############################################################################
class KMapperWrapper(BaseMapperWrapper):

    def __init__(
            self, 
            projection=None, scaler=None, 
            cover=None, clusterer=None, 
            remove_duplicate_nodes=False,
            memory='dyneusr_cache', 
            verbose=1
            ):
        """ Wraps KeplerMapper 

        Usage
        -----
            mapper = KMapperWrapper(projection=PCA(3), cover=dict(r=10, g=2))
            l = mapper.fit(X)
            g = mapper.map(l, X)

            # or 
            g = mapper.fit_map(X)
        """     
        try:
            from kmapper import KeplerMapper
            from kmapper.cover import Cover
        except ImportError as e:
            print("[warning]", e)

        # init mapper
        self.mapper = KeplerMapper()
        self.verbose = verbose

        # [1] fit params
        self.projection = projection if projection is not None else PCA(2)
        self.scaler = scaler or MinMaxScaler()

        # [2] map params
        self.clusterer = clusterer or DBSCAN(eps=1, min_samples=2)
        self.cover = cover or Cover(10, 0.5)
        self.remove_duplicate_nodes = remove_duplicate_nodes

        # setup memory
        self.memory = Memory(memory, verbose=verbose)


    def reset(self):
        self.data_ = None
        self.lens_ = None
        self.graph_ = None
        return self
             

    def fit_lens(self, data=None, projection=None, scaler=None, **kwargs):
        """ Fit a lens over data.
        """        
        # init params
        #self.mapper = KeplerMapper(verbose=self.verbose-1)
        self.projection = projection or self.projection 
        self.scaler = scaler or self.scaler

         # fit lens
        _transform_lens_cached = self.memory.cache(_transform_lens)
        lens = _transform_lens_cached(
            data=data,  
            projection=self.projection, 
            scaler=self.scaler,
            verbose=self.verbose
            )

        # save variables
        self.data_ = data
        self.lens_ = lens
        return self



    def fit_graph(self, lens=None, data=None, clusterer=None, cover=None, **kwargs):
        """ Fit a lens over data, map data into graph.
        """
        # extract inputs
        data = self.data_ if data is None else data
        lens = self.lens_ if lens is None else lens

        # init params
        #self.mapper = KeplerMapper(verbose=self.verbose)
        self.clusterer = clusterer or self.clusterer
        self.cover = cover or self.cover

        # fit graph
        _map_graph_cached = self.memory.cache(_map_graph)
        graph = _map_graph_cached(
            lens=lens, data=data,
            clusterer=self.clusterer,
            cover=self.cover,
            verbose=self.verbose, 
            remove_duplicate_nodes=self.remove_duplicate_nodes
            )

        # save variables
        self.data_ = data
        self.lens_ = lens
        self.graph_ = graph
        return self


    def fit(self, data=None, lens=None, **kwargs):
        """ Fit a lens over data, map data into graph.
        """
        # [1] fit lens
        if lens is None:
            self.fit_lens(data, **kwargs)
            lens = self.lens_
     
        # [2] map graph
        self.fit_graph(lens, data=data, **kwargs)
        return self


###############################################################################
### wrappers as functions
###############################################################################    
def fit_kmapper(data, **params):
    mapper = KMapperWrapper(**params)
    return mapper.fit(data=data)


def run_kmapper(data, **params):
    mapper = KMapperWrapper(**params)
    mapper.fit(data=data)
    # save as bunch
    result = Bunch(
        data=mapper.data_,
        lens=mapper.lens_,
        graph=mapper.graph_,
        params=params,
        )
    return result


    
   
   