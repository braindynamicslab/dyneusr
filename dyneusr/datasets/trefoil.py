"""
Trefoil knot data loader
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from itertools import product 

import numpy as np
import pandas as pd 

from sklearn.datasets.base import Bunch

import matplotlib as mpl
mpl.use('TkAgg', warn=False)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set("paper", "white")


##############################################################################
### data generators
##############################################################################
def make_trefoil(size=100, noise=0.0, a=2, b=2, c=3, **kwargs):
    """Generate synthetic trefoil dataset.

    Params
    -----
    :size = int (default = 1000)
     - the number of data points to use
    """
    # generate trefoil
    phi = np.linspace(0, 2*np.pi, size)
    x = np.sin(phi) + a*np.sin(a*phi)
    y = np.cos(phi) - b*np.cos(b*phi)
    z = -np.sin(c*phi)

    # add noise
    if noise is not None and noise > 0.0:
        noise = np.ravel(noise)
        if noise.shape[0] == 1:
            noise = noise.repeat(3)
        x += np.random.normal(0, noise[0], size)
        y += np.random.normal(0, noise[1], size)
        z += np.random.normal(0, noise[2], size)
    
    # remove tiny numbers
    x[np.abs(x) < 1e-6] = 0
    y[np.abs(y) < 1e-6] = 0
    z[np.abs(z) < 1e-6] = 0

    # stack features
    data = np.c_[x, y, z][:]
    data_ids = np.arange(phi.shape[0])

    # define target
    target = np.copy(phi)
    target = np.roll(target, int(size*0.05))

    # coloring based on target
    cmap = plt.get_cmap('brg', 3)
    norm = mpl.colors.Normalize(target.min(), target.max())
    coloring = cmap(norm(target))


    # format data bunch
    dataset = Bunch(
        data=data,
        feature_names=["x", "y", "z"],
        target=target,
        coloring=coloring,
        cmap=cmap, norm=norm,
        # other meta-data
        index=data_ids,
        domain=phi,
        )
    return dataset



##############################################################################
### data visualizers
##############################################################################
def draw_trefoil3d(x=None, y=None, z=None, c=None, s='z', ax=None, fig=None, view=(90, -90), **kwargs):
    """Plot trefoil knot.
    """   
    if c is None:
    	c = np.copy(z)
    cmap = plt.get_cmap(kwargs.get('cmap', "brg"), 3)
    norm = mpl.colors.Normalize(c.min(), c.max())

    # extract x, y, z
    if s == 'z':
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        s = zbins**2

    # combine features
    X = np.c_[x, y, z]

    # plot data
    fig, axes = plt.subplots(1, 3, figsize=(15,5),subplot_kw=dict(projection='3d'))

    # 3 views
    for ax_i, ax in enumerate(axes):

        if ax_i == 0:
            xcol, ycol, zcol = 0, 1, 2
        elif ax_i == 1:
            xcol, ycol, zcol = 0, 2, 1
        elif ax_i == 2:
            xcol, ycol, zcol = 1, 2, 0
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        ax.scatter(X[:,xcol], X[:,ycol], X[:,zcol], c=c, s=s, alpha=0.8, cmap=cmap, norm=norm)
        ax.set_xlabel(list('xyz')[xcol], fontweight='bold')
        ax.set_ylabel(list('xyz')[ycol], fontweight='bold')
        ax.set_zlabel(list('xyz')[zcol], fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        #if ax_i == 1:
        #    view = (view[0]+0, view[1]+90)
        #if ax_i == 2:
        #    view = (view[0]+90, view[1]+0)

        ax.set_title("view = {}".format(view))
        ax.view_init(*view)

    return axes



def draw_trefoil(x=None, y=None, z=None, c=None, s='z', ax=None, fig=None, **kwargs):
    """Plot trefoil knot.
    """   
    if c is None:
        c = np.copy(z)
    cmap = plt.get_cmap(kwargs.get('cmap', "brg"), 3)
    norm = mpl.colors.Normalize(c.min(), c.max())

    # extract x, y, z
    if s == 'z':
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        s = zbins**2

    # combine features
    X = np.c_[x, y, z]

    # plot data
    fig, axes = plt.subplots(3, 1, figsize=(15,5))

    # subplots of each dim as time-series
    for ax_i, (col_name, col_ys) in enumerate(dict(x=x, y=y, z=z).items()):
        ax = axes[ax_i]
        ax.scatter(np.arange(len(col_ys)), col_ys, c=c, s=s, cmap=cmap, norm=norm)
        ax.set_ylabel(col_name, fontweight='bold')
        if ax_i < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('index', fontweight='bold')
    return axes



