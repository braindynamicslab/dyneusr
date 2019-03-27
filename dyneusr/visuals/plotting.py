"""
Plotting wrappers and helper functions.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg', warn=False)
import matplotlib.pyplot as plt


import seaborn as sns 

from sklearn.preprocessing import Normalizer, Binarizer



def plot_temporal_degree(TCM, y=None, save_as=None, show=False, scalers=[Normalizer()], cmap='plasma', windows=None, **kwargs):
        """ Plot temporal transitions

        """
        def draw_axlines(*axline_funcs):
            """ Draw axlines based on y groups
            """
            if y is None:
                return 
            for i, y_ in enumerate(y):
                if i+1==y.shape[0] or y_ == y[i+1]:
                    continue
                for axline_func in axline_funcs:
                    axline_func(i, color="darkslategray")
            return

        # copy tcm
        tcm = np.copy(TCM)

        # scale ?
        #for scaler in filter(None, np.ravel(scalers)):
        #    tcm = scaler.fit_transform(tcm)
    
        figs = dict()

        ### heatmap of TCM      
        color = kwargs.get('color', 'steelblue')

        # setup figure, unless already exists
        fig = kwargs.get('fig')
        ax = kwargs.get('ax')
        if ax is not None:
            fig = ax.get_figure()
        elif fig is not None:
            ax = plt.gca()
        else:
            fig = plt.figure(figsize=(20,5))
            ax = plt.subplot2grid((5, 5), (0, 0), rowspan=5, colspan=5)

        ### tsplot
        #fig = plt.figure(figsize=(5,2))
        draw_axlines(plt.axvline)

        # normalized degree
        #deg = (tcm > 0.0).astype(float).sum(axis=1)
        deg = tcm.sum(axis=1)
        deg /= deg.max()
        
        # moving average
        window = kwargs.pop('window', 1) 
        deg_mva = deg.copy()
        if window > 1:
            rolling_kind = kwargs.get('rolling', np.mean)
            deg_rolling = pd.DataFrame(deg).rolling(window, center=True, min_periods=int(window/2))
            deg_mva = deg_rolling.apply(rolling_kind, raw=True)
            deg_mva /= deg_mva.max()
        # scale ?
        #deg_mva = deg_mva.reshape(-1, 1)
        #for scaler in filter(None, np.ravel(scalers)):
        #    deg_mva = scaler.fit_transform(deg_mva)

   
        # plot
        trs = np.arange(deg.shape[0])
        plt.plot(trs, np.ravel(deg),ls='-', marker='', alpha=0.3, color=color)
        plt.plot(trs, np.ravel(deg_mva)[:len(trs)], alpha=1.0,lw=3, color=color)

        ax = plt.gca()
        #ax.set_ylim(0.0, 1.1)
        xlim = kwargs.get('xlim', (trs.min(), trs.max()))
        ax.set_xlim(xlim)
        
        if xlim is not None:
            ax.set_xlim(xlim)

        title = kwargs.get('title') or 'Degree (TCM)'
        if title:
            ax.set_xlabel("Time frame (TR)", fontweight='bold', fontsize=16)
            ax.set_ylabel("Normalized degree", fontweight='bold', fontsize=16)
            ax.set_title(title, fontweight='bold', fontsize=18)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
        #figs['tsplot'] = fig
        if save_as is not None:
            print("Saving figure, save_as:", save_as)
            if not os.path.exists(os.path.dirname(save_as)):
                os.makedirs(os.path.dirname(save_as))
            fig.savefig(save_as, transparent=True)#, facecolor='lightgray', edgecolor='w')

        # show
        if show is True:
            plt.show()
        
        # save
        return fig, ax



# -----------------------------------------------------------
def plot_scatter(x, y, meta=None, fig=None, **scatter_kw):
    
    # figure
    fig = fig or plt.figure(figsize=(15,3))
    
    # update scatter_kw with colors
    if scatter_kw.get('c') is None:
        scatter_kw['c'] = meta.colors
    
    # take a look at the projection
    ax = fig.add_subplot(141)
    ax.scatter(x, y, **scatter_kw)
    ax.set_xlabel('Lens 1', fontweight='bold')
    ax.set_ylabel('Lens 2', fontweight='bold')
    
    # lets try again, this time in 3d (over time)
    from mpl_toolkits.mplot3d import Axes3D
    z = range(len(x))
    
    # second view
    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(z, x, y, **scatter_kw)
    ax.set_xlabel('TR', fontweight='bold')
    ax.set_ylabel('Lens 1', fontweight='bold')
    ax.set_zlabel('Lens 2', fontweight='bold')
    ax.view_init(5, 0)
    
    # second view
    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(y, z, x, **scatter_kw)
    ax.set_xlabel('Lens 2', fontweight='bold')
    ax.set_ylabel('TR', fontweight='bold')
    ax.set_zlabel('Lens 1', fontweight='bold')
    ax.view_init(5, 0)
    
    # first view
    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(x, z, y, **scatter_kw)
    ax.set_xlabel('Lens 1', fontweight='bold')
    ax.set_ylabel('TR', fontweight='bold')
    ax.set_zlabel('Lens 2', fontweight='bold')
    ax.view_init(5, 0)

    return fig


def plot_multiscatter(x, y, meta=None, **scatter_kw):
    figs = []
    for group_i, group in enumerate(sorted(meta.groups.unique())):
        group_meta = meta.loc[meta.groups.eq(group), :]
        fig = plot_scatter(
            x[group_meta.index.values], y[group_meta.index.values],
            meta=group_meta,
        )
        plt.title('Group {}'.format(group))
        figs.append(fig)
    return figs
