dyneusr
=======

DyNeuSR is a Python package for generating dynamical neuroimaging spatiotemporal representations using topological data analysis.


Important links
===============

- Official source code repo: https://bitbucket.org/braindynamicslab/dyneusr/
- Example notebooks repo: https://bitbucket.org/braindynamicslab/dyneusr-notebooks/


Dependencies
============

The required dependencies to use the software include:

* Python >= 3.5
* Numpy >= 1.15
* Pandas >= 0.23
* Scipy >= 1.0
* Scikit-learn >= 0.19
* Matplotlib >= 2.2
* NetworkX >= 2.2
* Nilearn >= 0.5

For a full list of requirements, see: `requirements.txt`


Install
=======

First, clone the repository by running the following command in a command prompt:
	
	
	git clone https://bitbucket.org/braindynamicslab/dyneusr/
	cd dyneusr
	

Next, to install the required dependencies, run the following command:


	pip install -r requirements.txt


Finally, to install `dyneusr` as a Python module, run the following command:
	

	pip install -e .


Usage
=====

Once installed, you can import `dyneusr` into any Python (3.5+) environment, using the following line of code:

	import dyneusr
