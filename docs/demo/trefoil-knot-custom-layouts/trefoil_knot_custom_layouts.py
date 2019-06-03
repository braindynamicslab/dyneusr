import networkx as nx
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from dyneusr import DyNeuGraph
from dyneusr.datasets import make_trefoil
from kmapper import KeplerMapper



# Generate synthetic dataset
dataset = make_trefoil(size=100)
X = dataset.data
y = dataset.target



# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=[0])
graph = mapper.map(lens, X, nr_cubes=6, overlap_perc=0.2)



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

# Visualize 
dG.visualize(static=True, show=True)
