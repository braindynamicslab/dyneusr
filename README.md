# DyNeuSR: **Dy**namical **Neu**roimaging **S**patiotemporal **R**epresentations

<p align="center">
<img src="https://raw.githubusercontent.com/braindynamicslab/dyneusr/master/docs/assets/logo.png" height="250">
</p>


DyNeuSR is a Python visualization library for topological representations of neuroimaging data. 

[DyNeuSR](https://braindynamicslab.github.io/dyneusr/) connects the Mapper algorithm (e.g., [KeplerMapper](https://kepler-mapper.scikit-tda.org)) with network analysis tools (e.g., [NetworkX](https://networkx.github.io/)) and other neuroimaging data visualization libraries (e.g., [Nilearn](https://nilearn.github.io/)). It provides a high-level interface for interacting with shape graph representations of neuroimaging data and relating such representations back to neurophysiology.

This package was designed specifically for working with shape graphs produced by the Mapper algorithm from topological data analysis (TDA) as described in the paper ["Towards a new approach to reveal dynamical organization of the brain using topological data analysis"](https://www.nature.com/articles/s41467-018-03664-4) (Saggar et al., 2018). See this [blog post](https://bdl.stanford.edu/blog/tda-cme-paper/) for more about the initial work that inspired the development of DyNeuSR.  




## Documentation

Online documentation will include detailed API documentation, Jupyter notebook examples and tutorials, and other useful information. (*coming soon*)

For now, Jupyter notebooks are available [here](https://github.com/braindynamicslab/dyneusr-notebooks/). These should correspond with the latest release of the master branch.
 


## Usage

The documentation will have an [example gallery](https://github.com/braindynamicslab/dyneusr/blob/master/examples/) with short Jupyter notebooks highlighting different aspects of DyNeuSR. (*coming soon*)

For more detailed examples, see these [notebook tutorials](https://github.com/braindynamicslab/dyneusr-notebooks/).

### Python code 

```python

import numpy as np 
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from dyneusr import DyNeuGraph

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_haxby()
masker = NiftiMasker(dataset.mask_vt[0], standardize=True)
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
y = pd.read_csv(dataset.session_target[0], sep=" ")
y = pd.DataFrame({_:y.labels.eq(_) for _ in np.unique(y.labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=PCA(3))
graph = mapper.map(lens, X, cover=Cover(10, 0.5), clusterer=DBSCAN(eps=30.))

# Visualize shape graph with DyNeuSR's DyNeuGraph 
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('dyneusr_output.html', show=True, port=8000)      

```





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
