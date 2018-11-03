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
        for scaler in filter(None, np.ravel(scalers)):
            tcm = scaler.fit_transform(tcm)
    
        figs = dict()

        ### heatmap of TCM      
        fig = kwargs.pop('fig', plt.figure(figsize=(20,5)))
        color = kwargs.get('color', 'steelblue')

        #ax = plt.subplot2grid((5, 5), (0, 0), rowspan=4, colspan=5)
        #sns.heatmap(tcm,
        #    square=True, cbar=False,
        #    xticklabels=100, yticklabels=100,
        #    cmap=cmap
        #    )
        #draw_axlines(plt.axvline, plt.axhline)

        # finish (e.g. label, etc)
        #ax = plt.gca()
        #ax.set_xlabel("Time frame (TR)")
        # ax.set_ylabel("Time frame (TR)")
        #ax.set_title('Temporal Connectivity Matrix (TCM)')
        #fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
        #figs['heatmap'] = fig
        #if save:
        #    save_as = 'TCM_heatmap.png'
        #    print("Saving figure, save_as:", save_as)
        #    plt.savefig(save_as, transparent=True)

        ### tsplot
        #fig = plt.figure(figsize=(5,2))
        ax = plt.subplot2grid((5, 5), (0, 0), rowspan=5, colspan=5)
        draw_axlines(plt.axvline)

        # normalized degree
        deg = (tcm > 0.0).astype(float).sum(axis=1)
        deg /= deg.max()
        
        
        # moving average
        window = kwargs.pop('window', 20) 
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
        plt.plot(trs, np.ravel(deg),ls='-', marker='', alpha=0.1, color=color)
        plt.plot(trs, np.ravel(deg_mva)[:len(trs)], alpha=1.0,lw=3, color=color)

        ax = plt.gca()
        #ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("Time frame (TR)", fontweight='bold', fontsize=16)
        ax.set_ylabel("Normalized degree", fontweight='bold', fontsize=16)
        ax.set_title('Degree (TCM)', fontweight='bold', fontsize=18)
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
        return save_as



# -----------------------------------------------------------




# -----------------------------------------------------------
from nilearn.image import iter_img
from nilearn.plotting import plot_prob_atlas, show

def plot_atlas(img):
    # Plot all ICA components together
    plot_prob_atlas(img, title='All ICA components')

    # Plot individual ICA components seperately
    for i, img_slice in enumerate(iter_img(img)):
        plot_prob_atlas(
            [img_slice], 
            title="Slice {}".format(i),
        )
    show()
    return


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
