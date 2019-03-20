from setuptools import find_packages, setup
import re

# parse dyneusr/_version.py
try:
    version_fn = 'dyneusr/_version.py'
    with open(version_fn) as version_fd:
        version = version_fd.read()
    version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
    version = re.findall(version_re, version, re.M)[0]
except:
    raise RuntimeError("Unable to read version in {}.".format(version_fn))


# parse requirements.txt
with open('requirements.txt') as f:
    install_requires = [_ for _ in f.read().split('\n') 
                        if len(_) and _[0].isalpha()]

# parse README.md
with open('README.md') as f:
    long_description = f.read()

# run setup
setup(
    name='dyneusr',
    version=version,
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
