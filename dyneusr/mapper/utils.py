""" 
Utility functions for running Mapper.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS 

###############################################################################
### Optional imports
###############################################################################
# UMAP
try:
    from umap.umap_ import UMAP
except ImportError as e:
    print("[warning]", e)

# HDBSCAN
try:
    from hdbscan import HDBSCAN
except ImportError as e:
    print("[warning]", e)
    

# External mapper tools
try:
    from kmapper import KeplerMapper
    from kmapper.cover import Cover
except ImportError as e:
    print("[warning]", e)
    

###############################################################################
### Helper functions
###############################################################################
def optimize_cover(X=None, r=30, g=3, limits=True, ndim=2):
    """ Get optimized cover for data.

    Notes
    -----
    - Requires kmapper

    """   
    from kmapper.cover import Cover
    
    # Define r, g based on data
    if X is not None:
        r = r * (len(X) / 1000.)

    # Get n_cubes, overlap
    n_cubes = int(max(1, r))
    p_overlap = float(g)

    # Convert to percent, if gain > 1
    if g > 1:
        p_overlap = (g-1) / float(g)
   
    # Define optimized limits
    if limits is True:
        offset = p_overlap / float(n_cubes)
        limits = [[-offset, 1+offset] for _ in range(ndim)]

    try:
        # Initialize Cover with limits
        cover = Cover(n_cubes, p_overlap, limits=limits)
    except Exception as e:
        # Ignore limits, probably using older version
        cover = Cover(n_cubes, p_overlap)
        print("[warning]", e)

    return cover



def optimize_dbscan(X, **kwargs):
    """ Get dbscan based on eps determined by data.
    """
    eps = optimize_eps(X)
    dbscan = DBSCAN(eps=eps, **kwargs)
    return dbscan



def optimize_eps(X, threshold=100, k=2, tree=None):
    """ Get optimized value for eps based on data.
    """
    from sklearn.neighbors import KDTree

    # Initialize neighbor tree
    if tree is None:
        # Use 'minkowski', p=2 (i.e. euclidean metric)
        tree = KDTree(X, leaf_size=30, metric='minkowski', p=2)

    # Query k nearest-neighbors for X
    dist, ind = tree.query(X, k=k)

    # Find eps s.t. % of points within eps of k nearest-neighbor 
    eps = np.percentile(dist[:, k-1], threshold)
    return eps
