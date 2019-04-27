import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from dyneusr import DyNeuGraph
from dyneusr.datasets import make_trefoil
from dyneusr.tools import visualize_mapper_stages
from kmapper import KeplerMapper

# Generate synthetic dataset
dataset = make_trefoil(size=100)
X = dataset.data
y = dataset.target

# Define projections to compare
projections = ([0], [0,1], [1,2], [0, 2])

# Compare different sets of columns as lenses
for projection in projections:

	# Generate shape graph using KeplerMapper
	mapper = KeplerMapper(verbose=1)
	lens = mapper.fit_transform(X, projection=projection)
	graph = mapper.map(lens, X, nr_cubes=4, overlap_perc=0.3)

	# Visualize the stages of Mapper
	fig, axes = visualize_mapper_stages(
		dataset, lens=lens, graph=graph, cover=mapper.cover, 
		layout="spectral", figsize=(16, 4))

	# Save each figure
	plt.savefig(
		"mapper_lens_{}.png".format("_".join(str(_) for _ in projection),
		dpi=600, background='transparent')

# Show all figures
plt.show()

