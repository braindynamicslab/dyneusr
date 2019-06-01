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
