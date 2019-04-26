import webbrowser

import numpy as np 
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from dyneusr import DyNeuGraph

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_haxby()
masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=2.5,
    memory="nilearn_cache"
    )
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
y = pd.read_csv(dataset.session_target[0], sep=" ")
y = pd.DataFrame({_:y.labels.eq(_) for _ in np.unique(y.labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=TSNE(2))
graph = mapper.map(
    lens, X=X, 
    cover=Cover(20, 0.5), 
    clusterer=DBSCAN(eps=20.)
    )

# Visualize the shape graph using DyNeuSR's DyNeuGraph
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_haxby_decoding.html', port=8800)   

# Explore/interact with the visualization in your browser
webbrowser.open(dG.HTTP.url)
