"""
Mixture model tools.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import scipy.stats

from sklearn.preprocessing import StandardScaler, Normalizer

from nilearn.input_data import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_msdl
from nilearn.connectome import ConnectivityMeasure
from nilearn import image
from nilearn import plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg', warn=False)



######################################################################
###
######################################################################
def mean_mixture(data, index=None, fwhm=None, threshold=None, **kwargs):
    # extract array
    try:
        X = data.data.values
    except:
        X = data.X
   
    # extract rows based on index
    mm_X = X[list(index), :].copy() # 
    mm_X = mm_X.mean(axis=0, keepdims=True)

    # unmask
    mm_img = data.masker.inverse_transform(mm_X)
  
    # smooth, threshold
    #mm_img = image.mean_img(mm_img)
    if fwhm is not None:
        mm_img = image.smooth_img(mm_img, fwhm=fwhm)        
    if threshold is not None:
        mm_img = image.threshold_img(mm_img, threshold)
    return mm_img


def simple_mixture(data, index=None, agg='mean', fwhm=None, threshold=None, **kwargs):
    # extract array
    try:
        X = data.data.values
    except:
        X = data.X
   
    # extract rows based on index
    mm_X = X[list(index), :].copy()
    mm_pos_X = np.copy(mm_X)
    mm_neg_X = np.copy(mm_X)

    # zero out pos, neg in opposite array
    mm_pos_X[(mm_X < 0.0)] = 0.0
    mm_neg_X[(mm_X > 0.0)] = 0.0

    # means
    # TODO: use StandardScaler().fit_transform(...)
    mean_pos_X = mm_pos_X.mean(axis=0) / (mm_pos_X.std(axis=0)+1) * len(index)
    mean_neg_X = mm_neg_X.mean(axis=0) / (mm_neg_X.std(axis=0)+1) * len(index)
    stack_pos_neg_X = np.stack([mean_pos_X, mean_neg_X])

    # agg over stack
    np_agg = agg if callable(agg) else getattr(np, agg)
    pos_neg_X = np_agg(stack_pos_neg_X, axis=0, keepdims=True)
    mm_X = mm_X.mean(axis=0, keepdims=True)#pos_neg_X.astype(np.float32)

    # unmask
    #mm_X = scipy.stats.zscore(mm_X, axis=-1)
    mm_img = data.masker.inverse_transform(mm_X)

    # clean
    #mm_img = image.clean_img(mm_img, standardize=True)

    # smooth, threshold
    if fwhm is not None:
        mm_img = image.smooth_img(mm_img, fwhm=fwhm)        
    if threshold is not None:
        mm_img = image.threshold_img(mm_img, threshold)
    return mm_img


def simple_mixtures(data, mixtures=[], targets=None, kind='simple', mode='glass', prefix='timestep', save_dir='tooltips', show_every_n=0, print_every_n=1, figsize=(4,4), plot_kws=dict(), **kwargs):
    # make sure output path exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check mixtures
    if isinstance(mixtures, dict):
        mixtures = mixtures.items()
    else:
        mixtures = list(enumerate(mixtures))
   
    # loop over mixtures
    filenames = []    
    for i, (id_, mixture) in enumerate(mixtures):
        # continue
        if not len(mixture):
            continue

        # format path to save figure
        save_as = os.path.join(save_dir, '{}_MM_{}{}.jpg'.format(kind, prefix, id_))
        filenames.append(save_as)  

        # title
        title = "{}{}".format(prefix, id_)
        try:
            target = targets[id_]
            title += " ({})".format(target)
        except:
            pass

    
        # plot, if file does not already exists
        if os.path.exists(save_as):

            # show the image, even if it already exists
            if show_every_n>0 and (i)%show_every_n == 0:
                plt.figure(figsize=figsize)
                plt.imshow(plt.imread(save_as))
                plt.axis('off')
                plt.show()
            continue

        # get simple mixture model
        mm = None
        if kind == 'mean':
            mm = mean_mixture(data, index=mixture, **kwargs)
        else:
            mm = simple_mixture(data, index=mixture, **kwargs)


        # plot simple mixture
        fig = plt.figure(figsize=figsize)
        if 'glass' in mode:
            # setup for plotting
            plot_kws = dict(dict(
                display_mode='z', plot_abs=False, 
                vmin=-3, vmax=3, #threshold=0.5,
                #cmap='jet', 
                colorbar=True,
                ), **plot_kws)
            # plot
            plotting.plot_glass_brain(mm, figure=fig, title=title, **plot_kws)
        else:
            # setup for plotting
            plot_kws = dict(dict(
                display_mode='z', cut_coords=1, 
                #threshold=0.5, 
                ), **plot_kws)
            # plot
            plotting.plot_stat_map(mm, figure=fig, title=title, **plot_kws)


        # save, show, close
        plt.savefig(save_as, dpi=300, transparent=True)
        if show_every_n>0 and (i)%show_every_n == 0:
            plt.show()
        plt.close('all')
      
        # display progress
        if print_every_n>0 and (i)%print_every_n == 0:
            print("[{} of {}] Saved: {}".format(i, len(mixtures), save_as))
    print('[done]')
    return filenames



######################################################################
###
######################################################################
def connectome_mixtures(data, mixtures=[], metric="correlation", prefix='', save_dir='figures', show=False, reset=False, **plot_kws):
    filenames = []
    if isinstance(mixtures, dict):
        mixtures = mixtures.items()
    else:
        mixtures = enumerate(mixtures)
    for id_, mixture in mixtures:
        print("Building Mixture Connectome... ID: {} (size = {})".format(id_, len(mixture)))
        save_as = 'CC_{}__{}{}.png'.format(metric, prefix, id_).replace(' ','_')
        save_as = os.path.join(save_dir, save_as)
        if not os.path.exists(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        if not os.path.exists(save_as) or reset is True:
            display = plot_connectome_mixture(data, index=mixture, metric=metric, save_as=save_as, show=show, **plot_kws)          
        # save filename
        filenames.append(save_as)
    return filenames



def plot_connectome_mixture(data=None, index=None, metric="correlation", save_as=None, show=False, **kwargs):
   
    # extract time series from all subjects and concatenate them
    mm = data.X[index, :].copy()
    time_series = [np.vstack(mm)]

    # calculate correlation matrices across indexed frames in data 
    connectome_measure = ConnectivityMeasure(kind=metric)
    connectome_measure.fit_transform(time_series)
    connectivity = connectome_measure.mean_
    np.fill_diagonal(connectivity, 0)
    #connectivity[np.abs(connectivity) < 0.2] = 0.0 


    # grab center coordinates for atlas labels
    atlas = kwargs.pop('atlas', data.atlas)
    coords = plotting.find_parcellation_cut_coords(labels_img=atlas)

    # assign node colors
    cmap = kwargs.pop('cmap', 'jet')
    node_cmap = plt.get_cmap('bone')
    node_norm = mpl.colors.Normalize(vmin=-0.8, vmax=1.2)
    node_colors = np.ravel([_[-1] for i,_ in enumerate(coords)])
    node_colors = [_ / np.max(node_colors) for _ in node_colors]
    node_colors = node_cmap(node_norm(node_colors))

    # plot connectome matrix
    fig = plt.figure(figsize=(12,5))
    ax = plt.subplot2grid((1, 2), (0, 1),  rowspan=1, colspan=1) 
    display = plotting.plot_matrix(
        connectivity,
        vmin=-.5, vmax=.5, colorbar=True, cmap=cmap,
        axes=ax, #title='{} Matrix'.format(metric.title()),
        )

    # plot connectome with 99.7% edge strength in the connectivity
    ax = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
    display = plotting.plot_connectome(
        connectivity, coords,
        edge_threshold="99.9%", display_mode='z',
        node_color=node_colors, node_size=20, edge_kwargs=dict(lw=4),
        edge_vmin=-.8, edge_vmax=.8, edge_cmap=cmap,
        colorbar=False, black_bg=not True, alpha=0.5,
        annotate=False,
        axes=ax,
        )
    if show is True:
        plt.show()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    if save_as:
        fig.savefig(save_as, transparent=True)#, facecolor='slategray', edgecolor='white')
    plt.close(fig)
    return display



def spatial_correlations(mm, atlas=None):
    if atlas is None:
        from nilearn.datasets import fetch_atlas_msdl
        atlas = fetch_atlas_msdl()
    from nilearn.input_data import NiftiMapsMasker
    masker = NiftiMapsMasker(maps_img=atlas['maps'])
    rsns_masked = masker.fit_transform(atlas['maps'])
    mm_masked = masker.fit_transform([mm])
    cc = np.corrcoef(mm_masked, rsns_masked)
    return cc


def run_jobs(jobs, multiproc=False):
    # 3. get results
    results = dict()
    if multiproc:     
        print("Running jobs... (multiproc={})".format(multiproc))
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count()-1)
        results = {_:pool.apply_async(eval, jobs[_]) for _ in jobs}
        results = {_:results[_].get() for _ in results}
        pool.close()
        pool.join()
    else:
        print("Running jobs... (multiproc={})".format(multiproc))
        for i, job in jobs.items():
            print("[job, i={}] {}".format(i, job))
            results[tr] = eval(job)
    # return
    return results
