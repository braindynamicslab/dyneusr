# DyNeuSR: **Dy**namical **Neu**roimaging **S**patiotemporal **R**epresentations
----------------------------------------------------------


<div class="row">
  <a href="https://bdl.stanford.edu/projects/dyneusr/Supplemental_Data_1.pdf">
      <img src="https://commons.wikimedia.org/wiki/File:Blue_Trefoil_Knot_Animated.gif" height="135" width="135">
  </a>
  <a href="https://bdl.stanford.edu/projects/dyneusr/Supplemental_Data_2.pdf">
      <img src="https://bdl.stanford.edu/projects/dyneusr/Supplemental_Movie_1.gif" height="135" width="135">
  </a>
  <a href="https://bdl.stanford.edu/blog/tda-cme-paper/">
      <img src="https://bdl.stanford.edu/images/blog/saggar-tda-mapper-cme.png" height="135" width="135">
  </a>
</div>


DyNeuSR is a Python visualization library for topological representations 
of neuroimaging data. It was designed specifically for working with shape 
graphs produced by the Mapper algorithm from topological data analysis (TDA) 
as described in the paper ["Towards a new approach to reveal dynamical 
organization of the brain using topological data analysis"](https://www.nature.com/articles/s41467-018-03664-4) 
(Saggar et al., 2018). See this [blog post](https://bdl.stanford.edu/blog/tda-cme-paper/) 
for more about the initial work that inspired the development of DyNeuSR.  

DyNeuSR connects the Mapper algorithm (e.g., [KeplerMapper](https://kepler-mapper.scikit-tda.org)), 
with network analysis tools (e.g., [NetworkX](https://networkx.github.io/)) 
and other neuroimaging data visualization libraries (e.g., [Nilearn](https://nilearn.github.io/)). 
It provides a high-level interface for interacting with shape graph 
representations of neuroimaging data and relating such representations 
back to neurophysiology.



## Documentation
----------------

Online documentation will include detailed API documentation, Jupyter notebook examples and tutorials, and other useful information. (*coming soon*)

For now, Jupyter notebooks are available [here](https://bitbucket.org/braindynamicslab/dyneusr-notebooks/). These should correspond with the latest release of the master branch.
 


## Usage
--------

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


### Examples

The documentation will have an [example gallery](https://bitbucket.org/braindynamicslab/dyneusr/examples/) with short Jupyter notebooks highlighting different aspects of DyNeuSR. (*coming soon*)

For more detailed examples, see these [notebook tutorials](https://bitbucket.org/braindynamicslab/dyneusr-notebooks/).




## Install
----------

### Dependencies

- Python 3.6+

The latest versions of the following packages are required:

-  [numpy](www.numpy.org)

-  [pandas](pandas.pydata.org)

-  [scipy](www.scipy.org)

-  [matplotlib](matplotlib.sourceforge.net)

-  [seaborn](stanford.edu/~mwaskom/software/seaborn)

-  [networkx](networkx.github.io)

-  [scikit-learn](scikit-learn.org)

-  [nilearn](nilearn.github.io)

-  [kmapper](kepler-mapper.scikit-tda.org)

For the full list, see [`requirements.txt`](https://bitbucket.org/braindynamicslab/dyneusr/requirements.txt)



### Environment

The current implementation of `dyneusr` requires Python 3.6+. If your default environment is Python 2, we recommend that you install `dyneusr` in a separate Python 3 environment. 

```bash
conda create -n py36 python=3.6
```

To activate the environment, just do:

```bash
source activate py36
```


If you don't have conda, or are new to scientific python, we recommend that you download the [Anaconda scientific python distribution](https://store.continuum.io/cshop/anaconda/). You can find more information about creating a separate environment for Python 3, [here](https://salishsea-meopar-docs.readthedocs.io/en/latest/work_env/python3_conda_environment.html). 



### Installation

To get the `dyneusr` source code, just clone the Bitbucket repository:

```bash
git clone https://bitbucket.org/braindynamicslab/dyneusr.git
```


Enter the `dyneusr` directory:

```bash
cd dyneusr
```


To install all required packages, just do:

```bash
pip install -r requirements.txt
```


Finally, to install `dyneusr` as a Python module, just do:

```bash
pip install -e .
```


To instead use the development version from Bitbucket, by running:

```bash
pip install git+git://braindynamics.org/dyneusr/dyneusr.git#egg=dyneusr
```




## Development
--------------

All development happens here, on [Bitbucket]().

If you're interested in contributing to DyNeuSR, please refer to the [Contributing]() guide. (*coming soon*)




## Support
----------

Please [submit](https://bitbucket.org/braindynamicslab/dyneusr/issues/new) any bugs or questions to the Bitbucket issue tracker.



## License
----------

Released under a BSD-3 license



## Cite
-------

(*coming soon*)