from setuptools import find_packages, setup

# parse dyneusr/_version.py
with open('dyneusr/_version.py') as f:
    version = f.read().split('__version__ =')[-1]
    version = version.replace("'",'"').split('"')[1]

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
