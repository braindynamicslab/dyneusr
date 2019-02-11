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
def optimize_cover(X=None, r=30, g=0.67, scale_r=False, scale_g=False, ndim=2, scale_limits=True):
    """ Get optimized cover for data.

    Notes
    -----
    - Requires kmapper

    """   
    from kmapper.cover import Cover

    # Define r, g based on data / heuristic
    if X is not None:
        # Heuristic based on size, dimensionality of data
        scale_factor = (len(X) / 1000.) * (2. / ndim)
        
        # Scale r
        if scale_r:
            r = r * scale_factor

        # Scale g
        if scale_g:
            # Convert to gain, if defined as percent
            if g < 1:
                g = 1. / (1. - g)
            # Scale
            g = g * scale_factor


    # Convert gain to percent
    if g >= 1:
        g = (g - 1) / float(g)
        
    # Get n_cubes, overlap
    n_cubes = max(1, r)
    p_overlap = float(g)


    # Round final values 
    n_cubes = int(n_cubes)
    p_overlap = np.round(p_overlap, 2)

    # Define optimized limits
    limits = None
    if scale_limits is True:
        offset = p_overlap / float(n_cubes)
        limits = [[-offset, 1+offset] for _ in range(ndim)]
        n_cubes += 2

    try:
        # Initialize Cover with limits
        cover = Cover(n_cubes, p_overlap, limits=limits)
    except Exception as e:
        # Ignore limits, probably using older version
        cover = Cover(n_cubes, p_overlap)
        print("[warning]", e)
    return cover



def optimize_dbscan(X, n_neighbors=1, min_samples=2, metric='minkowski', leaf_size=30, p=2, **kwargs):
    """ Get dbscan based on eps determined by data.
    """
    eps = optimize_eps(X, n_neighbors=n_neighbors, metric=metric, leaf_size=leaf_size, p=p)
    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, 
        metric=metric, leaf_size=leaf_size, p=p, 
        n_jobs=-1, **kwargs
        )
    return dbscan



def optimize_eps(X, threshold=100, n_neighbors=1, metric='minkowski', leaf_size=30, p=2):
    """ Get optimized value for eps based on data.
    """
    from sklearn.neighbors import KDTree

    # Use 'minkowski', p=2 (i.e. euclidean metric)
    tree = KDTree(X, leaf_size=leaf_size, metric=metric, p=p)

    # Query k nearest-neighbors for X
    dist, ind = tree.query(X, k=n_neighbors+1)

    # Find eps s.t. % of points within eps of k nearest-neighbor 
    eps = np.percentile(dist[:, n_neighbors], threshold)
    return eps




