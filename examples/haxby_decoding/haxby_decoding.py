import webbrowser
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from dyneusr import DyNeuGraph
from dyneusr.tools import visualize_mapper_stages

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_haxby()
masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=2.5,
    memory="nilearn_cache"
    )
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
df = pd.read_csv(dataset.session_target[0], sep=" ")
target, labels = pd.factorize(df.labels.values)
y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=TSNE(2))
graph = mapper.map(
    lens, X=X, 
    cover=Cover(20, 0.5), 
    clusterer=DBSCAN(eps=20.)
    )

# Visualize the stages of Mapper
fig, axes = visualize_mapper_stages(
	dataset, y=target, lens=lens, graph=graph, cover=mapper.cover, 
	node_size=20, edge_size=0.5, edge_color='gray',
	layout="kamada_kawai",  figsize=(16, 3)
	)
plt.savefig("mapper_stages.png")
plt.show()

# Visualize the shape graph using DyNeuSR's DyNeuGraph
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_haxby_decoding.html', port=8800)   
webbrowser.open(dG.HTTP.url)
