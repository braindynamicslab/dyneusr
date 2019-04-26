# DyNeuSR: **Dy**namical **Neu**roimaging **S**patiotemporal **R**epresentations

<p align="center">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/docs/assets/logo.png" height="250">
</p>

DyNeuSR is a Python visualization library for topological representations of neuroimaging data. 

[DyNeuSR](https://braindynamicslab.github.io/dyneusr/) connects the Mapper algorithm (e.g., [KeplerMapper](https://kepler-mapper.scikit-tda.org)) with network analysis tools (e.g., [NetworkX](https://networkx.github.io/)) and other neuroimaging data visualization libraries (e.g., [Nilearn](https://nilearn.github.io/)). It provides a high-level interface for interacting with shape graph representations of neuroimaging data and relating such representations back to neurophysiology.

<p align="center">
<img src="https://raw.github.com/braindynamicslab/dyneusr/master/examples/haxby_decoding/mapper_stages.png">
</p>

This package was designed specifically for working with shape graphs produced by the Mapper algorithm from topological data analysis (TDA) as described in the paper ["Towards a new approach to reveal dynamical organization of the brain using topological data analysis"](https://www.nature.com/articles/s41467-018-03664-4) (Saggar et al., 2018). See this [blog post](https://bdl.stanford.edu/blog/tda-cme-paper/) for more about the initial work that inspired the development of DyNeuSR. 

<p align="center"><a href="https://github.com/braindynamicslab/dyneusr/blob/master/examples/haxby_decoding/haxby_decoding.py">
<img src="https://raw.github.com/braindynamicslab/dyneusr/master/examples/haxby_decoding/dyneusr_haxby_decoding.png">
</a></p>

## Documentation

Online documentation will include detailed API documentation, Jupyter notebook examples and tutorials, and other useful information. (*coming soon*)

For now, Jupyter notebooks are available [here](https://github.com/braindynamicslab/dyneusr-notebooks/). These should correspond with the latest release of the master branch.
 


## Examples

The documentation will include several [examples](https://github.com/braindynamicslab/dyneusr/blob/master/examples/) that introduce and highlight different aspects of DyNeuSR. 

For more detailed tutorials, see the [dyneusr-notebooks](https://github.com/braindynamicslab/dyneusr-notebooks/).


### Basic usage ([trefoil knot](https://github.com/braindynamicslab/dyneusr/blob/master/examples/trefoil_knot))


```python

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
fig, axes = visualize_mapper_stages(
	dataset, y=y, lens=lens, graph=graph, cover=mapper.cover, 
	layout="spectral")
 
# Visualize the shape graph using DyNeuSR's DyNeuGraph                          
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_trefoil_knot.html')

```


### Mapper comparisons ([trefoil knot](https://github.com/braindynamicslab/dyneusr/blob/master/examples/trefoil_knot))

```python
# Visualize the stages of Mapper
fig, axes = visualize_mapper_stages(
	dataset, y=y, lens=lens, graph=graph, cover=mapper.cover, 
	layout="spectral")
```

<p align="center"><a href="https://github.com/braindynamicslab/dyneusr/blob/master/examples/trefoil_knot">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/examples/trefoil_knot/mapper_lens_0.png">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/examples/trefoil_knot/mapper_lens_0_1.png">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/examples/trefoil_knot/mapper_lens_0_2.png">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/examples/trefoil_knot/mapper_lens_1_2.png">
</a></p>



### Neuroimaging examples ([haxby decoding](https://github.com/braindynamicslab/dyneusr/blob/master/examples/haxby_decoding))

```python
# Visualize the shape graph using DyNeuSR's DyNeuGraph                          
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_output.html')
```

<p align="center"><a href="https://github.com/braindynamicslab/dyneusr/blob/master/examples/haxby_decoding/haxby_decoding.py">
<img src="https://raw.github.com/braindynamicslab/dyneusr/master/examples/haxby_decoding/dyneusr_haxby_decoding.png">
</a></p>





## Setup

### Dependencies

- Python 3.6

The following Python packages are required:

-  [numpy](www.numpy.org)
-  [pandas](pandas.pydata.org)
-  [scipy](www.scipy.org)
-  [scikit-learn](scikit-learn.org)
-  [matplotlib](matplotlib.sourceforge.net)
-  [seaborn](stanford.edu/~mwaskom/software/seaborn)
-  [networkx](networkx.github.io)
-  [nilearn](nilearn.github.io)
-  [kmapper](kepler-mapper.scikit-tda.org)

For the full list of packages and required versions, see [`requirements.txt`](https://github.com/braindynamicslab/dyneusr/blob/master/requirements.txt) and [`requirements-versions.txt`](https://github.com/braindynamicslab/dyneusr/blob/master/requirements-versions.txt)



### Environment

If your default environment is Python 2, we recommend that you install `dyneusr` in a separate Python 3 environment. 

To create a new environment and activate it:
```bash
conda create -n py36 python=3.6
source activate py36
```

You can find more information about creating a separate environment for Python 3, [here](https://salishsea-meopar-docs.readthedocs.io/en/latest/work_env/python3_conda_environment.html). 

If you don't have conda, or are new to scientific python, we recommend that you download the [Anaconda scientific python distribution](https://store.continuum.io/cshop/anaconda/). 



### Installation

To download the source:
```bash
git clone https://github.com/braindynamicslab/dyneusr.git
cd dyneusr
```

To install from source:
```bash
pip install -e .
```



## Development

All development happens here, on [GitHub](https://github.com/braindynamicslab/dyneusr/). Please feel free to report any issues and propose improvements. 

If you're interested in contributing to DyNeuSR, please also refer to the [Contributing](https://github.com/braindynamicslab/dyneusr/blob/master/CONTRIBUTING.md) guide. 



## Support

Please [submit](https://github.com/braindynamicslab/dyneusr/issues/new) any bugs or questions to the GitHub issue tracker.



## License

Released under a BSD-3 license



## Citing DyNeuSR

>Geniesse C, Sporns O, Petri G, & Saggar M. [Generating dynamical neuroimaging spatiotemporal representations (DyNeuSR) using topological data analysis](). _In press_, 2019.
