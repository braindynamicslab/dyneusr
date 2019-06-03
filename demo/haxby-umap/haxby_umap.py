import numpy as np 
import pandas as pd
import networkx as nx

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from dyneusr import DyNeuGraph
from dyneusr.tools import visualize_mapper_stages
from dyneusr.mapper.utils import optimize_dbscan



# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_haxby()
masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=2.5,
    memory="nilearn_cache")
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
df = pd.read_csv(dataset.session_target[0], sep=" ")
target, labels = pd.factorize(df.labels.values)
y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})

# Extract sessions 4-5
mask_sessions = df.chunks.add(1).isin([4, 5])
X = X[mask_sessions]
y = y.loc[mask_sessions, :]
target = target[mask_sessions]



# Generate a shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)

# Configure projection
pca = PCA(2, random_state=1)
umap = UMAP(n_components=2, init=pca.fit_transform(X))

# Construct lens and generate the shape graph
lens = mapper.fit_transform(
	umap.fit_transform(X, y=None), 
	projection=[0, 1]) 
graph = mapper.map(
    lens, X=X, 
    cover=Cover(20, 0.5),
    clusterer=optimize_dbscan(X, k=3, p=100.0), )



# Convert to a DyNeuGraph
dG = DyNeuGraph(G=graph, y=y)

# Define some custom_layouts
dG.add_custom_layout(lens, name='lens')
dG.add_custom_layout(nx.spring_layout, name='nx.spring')
dG.add_custom_layout(nx.kamada_kawai_layout, name='nx.kamada_kawai')
dG.add_custom_layout(nx.spectral_layout, name='nx.spectral')
dG.add_custom_layout(nx.circular_layout, name='nx.circular')

# Configure some projections
pca = PCA(2, random_state=1)
tsne = TSNE(2, init='pca', random_state=1)
umap = UMAP(n_components=2, init=pca.fit_transform(X))

# Add projections as custom_layouts
dG.add_custom_layout(pca.fit_transform(X), name='PCA')
dG.add_custom_layout(tsne.fit_transform(X), name='TSNE')
dG.add_custom_layout(umap.fit_transform(X, y=None), name='UMAP')
dG.add_custom_layout(umap.fit_transform(X, y=target), name='Supervised UMAP')

# Visualize 
dG.visualize(static=True, show=True)