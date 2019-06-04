import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from dyneusr import DyNeuGraph
from dyneusr.tools import visualize_mapper_stages

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
df = pd.read_csv(dataset.session_target[0], sep=" ")
target, labels = pd.factorize(df.labels.values)
y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=TSNE(2, random_state=1))
graph = mapper.map(
    lens, X=X, 
    cover=Cover(20, 0.5), 
    clusterer=DBSCAN(eps=20.)
    )
dG = DyNeuGraph(G=graph, y=y)

# Define some custom_layouts
import networkx as nx
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
dG.add_custom_layout(nx.spring_layout, name='nx.spring')
dG.add_custom_layout(nx.kamada_kawai_layout, name='nx.kamada_kawai')
dG.add_custom_layout(nx.spectral_layout, name='nx.spectral')
dG.add_custom_layout(nx.circular_layout, name='nx.circular')
dG.add_custom_layout(PCA(2).fit_transform(X), name='PCA')
dG.add_custom_layout(TSNE(2).fit_transform(X), name='TSNE')

# Visualize
dG.visualize('dyneusr_custom_layouts.html', static=True, show=True)
