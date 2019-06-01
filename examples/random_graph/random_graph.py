from dyneusr import DyNeuGraph
from dyneusr.datasets import make_random_graph
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

# Visualize the shape graph using DyNeuSR's DyNeuGraph 
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_random_graph.html', static=True, show=True)  
