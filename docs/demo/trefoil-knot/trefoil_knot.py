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



# Convert to a DyNeuGraph, visualize
dG = DyNeuGraph(G=graph, y=y) 
dG.visualize(static=True, show=True)
