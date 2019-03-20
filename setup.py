from setuptools import find_packages, setup

# parse requirements.txt
with open('requirements.txt') as f:
    install_requires = [_ for _ in f.read().split('\n') if len(_) and _[0].isalpha()]

# parse README.md
with open('README.md') as f:
    long_description = f.read()

# run setup
setup(
    name='dyneusr',
    version='0.2.2',
    description='Dynamical Neural Spatiotemporal Representations.',
    long_description=long_description,
    long_description_content_type="text/markdown",	
    author='Caleb Geniesse',
    author_email='geniesse@stanford.edu',
    url='https://bitbucket.org/braindynamicslab/dyneusr/wiki/Home',
    license='BSD-3',
    packages=find_packages(),
	install_requires=install_requires,
    python_requires='>=3.4',
)
