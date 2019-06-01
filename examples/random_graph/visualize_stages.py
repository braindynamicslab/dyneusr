import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from dyneusr import DyNeuGraph
from dyneusr.datasets import make_random_graph
from dyneusr.tools import visualize_mapper_stages
from kmapper import KeplerMapper

# Generate synthetic dataset
dataset = make_random_graph(
	n_nodes=10, n_edges=10,
    n_features=10, n_local_samples=10, 
    random_state=None, verbose=1
)
X = dataset.data
y = dataset.target

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=[0])
graph = mapper.map(lens, X, nr_cubes=20, overlap_perc=0.67)

# Visualize the stages of Mapper
fig, axes = visualize_mapper_stages(
	dataset, y=y, lens=lens, graph=graph, cover=mapper.cover, 
	layout="spectral", figsize=(16, 4))
plt.savefig("mapper_stages.png", dpi=600, background="transparent")
plt.show()

