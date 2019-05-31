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

# Visualize the shape graph using DyNeuSR's DyNeuGraph 
dG = DyNeuGraph(G=graph, y=y)
<<<<<<< HEAD
dG.visualize('dyneusr_trefoil_knot.html', static=True, show=True)  
=======
dG.visualize('dyneusr_trefoil_knot.html', static=not True, show=True, port=8800)  
>>>>>>> 29af00ab7e8989084ad7d66e1e996d6daa799baa
