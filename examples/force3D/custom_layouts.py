import numpy as np
np.random.seed(123)
from dyneusr import DyNeuGraph
from dyneusr.datasets import make_trefoil
from kmapper import KeplerMapper
from sklearn.decomposition import PCA

# Generate synthetic dataset
import tadasets
X = tadasets.sphere(n=500, r=1)

# Sort by first column
inds = np.argsort(X[:, 0])
X = X[inds].copy()
y = np.arange(X.shape[0])

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=PCA(2))
graph = mapper.map(lens, X, nr_cubes=6, overlap_perc=0.5)
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
