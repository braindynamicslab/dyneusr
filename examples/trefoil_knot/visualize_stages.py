import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from dyneusr import DyNeuGraph
from dyneusr.datasets import make_trefoil
from dyneusr.tools import visualize_mapper_stages
from kmapper import KeplerMapper

# Generate synthetic dataset
dataset = make_trefoil(size=100)
X = dataset.data
y = dataset.target

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=[0, 1])
graph = mapper.map(lens, X, nr_cubes=4, overlap_perc=0.3)

# Visualize the stages of Mapper
_ = visualize_mapper_stages(
	dataset, lens=lens, 
	graph=graph, cover=mapper.cover, 
	node_size=300, edge_size=0.5, edge_color='gray',
	layout="spectral",  figsize=(16, 4),
	)

# Show 
plt.show()

