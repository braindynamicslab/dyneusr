"""
Graph processing and conversion tools.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools
import functools
import json
import os

from IPython.core.display import HTML, display

from collections import defaultdict, Counter

import multiprocessing as mp

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats

from sklearn.preprocessing import Normalizer, LabelEncoder
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt

def in_notebook():
    """ Returns ``True`` if the module is running in IPython kernel,
      ``False`` if in IPython shell or other Python shell.
    """
    from IPython import get_ipython
    return get_ipython() is not None


def _agg_proportions(df, members=None):
    """ Aggregate proportions df for members. 
    """
    p = df.copy()
    if members is not None:
        p = p.iloc[members]
    p = p.T.assign(
        group=pd.factorize(p.columns)[0],
        label=pd.factorize(p.columns)[-1],
        value=p.sum(), #/ p.sum().sum() * p.shape[0],
        row_count=p.shape[0]
        )
    p = p[['label', 'group', 'value', 'row_count']]
    p.columns = ['label', 'group', 'value', 'row_count']
    p = list(p.T.to_dict().values())
    return p


def process_meta(meta_, labels=None, zscore=True, **kwargs):
 
    # process each column of meta
    meta_sets = dict()
    meta_labels = dict()
    for i,meta_col in enumerate(meta_.columns):
        #print("Processing meta column: {}".format(meta_col))
        meta = np.ravel(meta_[meta_col].values.copy())
        if len(meta) < 1:
            continue
        # process meta label
        meta_label = None
        if isinstance(labels, dict):
            # one list per column
            if meta_col in labels:
                meta_label = list(labels[meta_col])
        elif isinstance(labels, list):
            # shared list
            if not isinstance(labels[0], list):
                meta_label = list(labels)

        # process meta
        if str(meta[0]).isalpha() or type(meta[0]) is str:
            encoder = LabelEncoder()
            yi = encoder.fit_transform(meta)
            yi_bins = np.linspace(yi.min(), yi.max(), num=min(5, len(set(yi))), endpoint=True)
            meta = np.digitize(yi, yi_bins, right=True)
            meta = yi_bins[meta]
            meta_label = [list(yi_bins).index(_) for _ in sorted(set(meta))]
            #meta_label = ['Group '+str(_+1) for _ in meta_label]
            meta_bins = [_ for _ in zip(yi_bins[:-1], yi_bins[1:])] 
            meta_label = ['Group {} {}'.format(_+1, __) for _,__ in enumerate(meta_bins)]

        
        elif len(set(meta)) > 9 and zscore is True:
            # TODO: how do we decide whether to use continuous vs. discrete
            # zscore
            yi = meta.copy()
            yi_nz = yi[~np.isnan(yi)]
            zi = stats.zscore(yi_nz)
            zi = np.sign(zi) * np.floor(np.abs(zi))

            # digitize
            zi_bins = np.arange(zi.min(), zi.max()+1, step=1)
            zi = np.digitize(zi, zi_bins, right=True) + 1
            yi[~np.isnan(yi)] = zi
            yi[np.isnan(yi)] = 0
            yi = yi.astype(int)
            meta_bins = np.ravel([np.nan] + list(zi_bins))
            meta_label = meta_bins[np.sort(np.unique(yi))]
            meta = yi.copy()

            # map meta to labels
            #meta_str = {False: '\u03BC (= {:0.2f})'.format(np.mean(yi_nz)),
            #            True: '\u03BC + {:1.0f}\u03C3'}
            meta_str = {False: 'Mean ({:0.2f})'.format(np.mean(yi_nz)),
                        True: '{:1.0f} S.D.'}
            meta_label = [meta_str[np.abs(float(_)) > 0].format(float(_)) for _ in meta_label if not np.isnan(_)]
            
            # add n/a label if nans found
            if len(yi_nz) < len(yi):
                meta_label = ['NaN'] + meta_label
        
        elif len(set(meta)) > 9 and zscore is False:
            # TODO: figureout continuous scale here
            pass

        # labels for legend
        meta_sets[meta_col] = [_ for _ in np.sort(np.unique(meta))]
        if meta_label is not None:
            meta_labels[meta_col] = [_ for _ in meta_label]
        #print("  [+] found {} unique groups.".format(len(meta_sets[meta_col])))

        # re-assign meta
        meta_[meta_col] = meta.copy()

    return meta_, meta_sets, meta_labels


def process_graph(graph=None, meta=None, tooltips=None, color_by=None, labels=None, **kwargs):
    # convert to nx
    if isinstance(graph, nx.Graph):
        g = graph.copy()
        graph = nx.node_link_data(g)
    elif graph is None:
        graph = {}

    # copy graph, override defaults
    graph = dict(dict(nodes={}, links={}), **graph)

    # return empty graph
    if len(graph['nodes']) + len(graph['links']) < 1:
        return nx.Graph()

    # extract node, links
    nodelist = dict(graph.get('nodes', {}))
    edgelist = dict(graph.get('links', {}))
    memberlist = np.unique([_ for n,d in nodelist.items() for _ in d])
    nTR = int(max(np.r_[len(memberlist), memberlist+1]))
    
    # index
    node_to_index = {n:i for i,n in enumerate(nodelist)}
    index_to_node = {i:n for i,n in enumerate(nodelist)}

    # meta stuff
    if meta is None:
        meta = pd.DataFrame(index=np.arange(nTR)).assign(
            default=0,
            )
    elif not isinstance(meta, pd.DataFrame):
        meta = meta.copy()
        meta = meta.reshape(len(meta), -1)
        # check labels
        columns = ['meta-column-{}'.format(i) for i,_ in enumerate(meta.T)]
        if labels is not None and len(labels):
            columns[:len(labels)] = list(labels.keys())
        # convert to DataFrame
        meta = pd.DataFrame(
            meta, 
            columns=columns
            )
        
    # save un-processed metadata
    meta = meta.copy()
    meta_orig = meta.copy()

    # add some defaults
    meta['data_id'] = np.arange(len(meta)).astype(str)
    meta['uniform'] = '0' 


    # normalize meta
    # TODO: move all of this logic into color map utils
    #meta = Normalizer().fit_transform(meta.reshape(-1, 1))
    # bin meta (?)
    meta, meta_sets, meta_labels = process_meta(meta, labels=labels, **kwargs)
    
    # color_by
    if len(meta_orig.T) < 2:        
        color_by = color_by or meta_orig.columns[0] 

    # multiclass proportions
    # TODO: make sure this works on edge cases
    multiclass = _agg_proportions(meta_orig)
    color_by = color_by or 'multiclass' 
    meta_sets['multiclass'] = [_.get('group') for _ in multiclass]
    meta_labels['multiclass'] = [_.get('label') for _ in multiclass]
    df_multiclass = pd.DataFrame(multiclass)
    tooltip = df_multiclass.to_html(
                        index=False, columns=['label','group','value','row_count'],
                        float_format='{:0.2f}'.format,
                        )
    if kwargs.get('verbose', 1) > 0:
        if in_notebook():
            display(HTML(tooltip))
        else:
            print(df_multiclass)

    # color functions
    color_functions = kwargs.get('color_functions', {})
    #if color_by not in color_functions:
    for color_by_ in meta_sets:
        # get hex color for each group
        color_values = meta_sets[color_by_]
        n_colors = len(np.unique(color_values)) 
        if n_colors < 3:
            continue

        # check cmap       
        cmap = kwargs.get('cmap', 'tab20c')
        if not callable(cmap):
            cmap = plt.get_cmap(cmap)

        # save color values (normalized)
        if n_colors < 20:
            norm = mpl.colors.Normalize(np.min(color_values), np.max(color_values))
            color_values = norm(color_values)        

        # save color function as hex
        color_function = [mpl.colors.to_hex(_) for _ in cmap(color_values)]
        color_functions[color_by_] = color_function

    # tooltips (TODO: should this be here)
    if tooltips is None:
        tooltips = np.array([
            "<b>{}</b> (<b>size:</b> {})<br><br>".format(
                k, len(_)
            ) for k,_ in nodelist.items()
            ])
    tooltips = np.array(tooltips).astype(str)


    ### NODES
    G = nx.Graph(
        labels=meta_labels,
        groups=meta_sets,
        color_by=color_by,
        color=color_functions,
        )
    for node_id, (name, members) in enumerate(nodelist.items()):
        # define node_dict for G
        members = list(sorted(members))
        tooltip = tooltips[node_id]

        # aggregate proportions into a single column
        multiclass = _agg_proportions(meta_orig, members)
        tooltip += pd.DataFrame(multiclass).to_html(
                        index=False, header=False, columns=['label','value'],
                        float_format='{:0.2f}'.format,
                        )

        proportions = dict(
            multiclass=multiclass
            )

        # format node dict
        node_dict = dict(
            id=int(node_id),
            name=name,
            tooltip=tooltip,
            members=members,
            proportions=proportions,
            group=-1,
            # node color, size
            size=len(members),
            ) 

        coloring = kwargs.get('rsn_color','separate')
        # proportions by column
        if coloring is 'separate': # Color networks separately or put all together in one graph
            group = dict()
            for meta_col in meta.columns:
                groups = meta[meta_col].values[members] 
                proportions[meta_col] = [
                    dict(group=int(_), row_count=len(groups), value=int(c))
                    for _,c in Counter(groups).most_common() #metaset
                    ]
                group[meta_col] = int(Counter(groups).most_common()[0][0])
                # format node dict
                node_dict.update(
                    proportions=proportions, 
                    group=group
                    )  

        elif coloring is 'together':
            # multilabel
            groups = meta.values[members, :-2] 
            allgroups = [np.nonzero(memlabels) for memlabels in groups]
            allgroups = [el for subarr in allgroups for subsubarr in subarr for el in subsubarr]   
            proportions['multi'] = [
                    dict(group=int(_), row_count=len(allgroups), value=int(c))
                    for _,c in Counter(allgroups).most_common() 
                    ]
            group = int(Counter(allgroups).most_common()[0][0])
            # format node dict
            node_dict.update(
                proportions=proportions, 
                group=group
                )  
        
        # update G
        G.add_node(name, **node_dict)
        
    ### EDGES
    for (source, targets) in edgelist.items():
        # add edge for each target
        for target in targets:
            source_id = node_to_index[source]
            target_id = node_to_index[target]

            source_node = G.node[source]
            target_node = G.node[target]

            # add directional edges
            s_members = np.sort(source_node['members'])
            t_members = np.sort(target_node['members'])

            # temporal connections (i.e. (t, t+1))
            members = set(s_members).intersection(t_members)
            st_members = set(s_members + 1).intersection(t_members) - members
            ts_members = set(t_members + 1).intersection(s_members) - members

            # add forward edge
            #if len(st_members):
            #    G.add_edge(source, target, 
            #        value=len(st_members),
            #        size=len(st_members),
            #        kind='directed'
            #    )

            # add reverse edge
            #if len(ts_members):
            #    G.add_edge(target, source, 
            #        value=len(ts_members),
            #        size=len(ts_members),
            #        kind='directed'
            #    )
            # compute direction (net flow)
            n_directional = len(st_members) - len(ts_members)
            direction = np.min(t_members) - np.min(s_members)

            # define edge dict
            edge_dict = dict(
                value=len(members),
                size=len(members),
                kind='intersection',
                n_directional=n_directional
                # edge color, size
            )

            # update G
            if ((n_directional < 0) or
                (n_directional == 0 and direction < 0)):
                # net flow target to source
                G.add_edge(target, source, **edge_dict)
            else:
                # net flow source to target
                G.add_edge(source, target, **edge_dict)
                


    # more attribues
    for n, nbrs in G.adj.items():
        G.node[n]['degree'] = G.degree(n)
        for nbr in nbrs:
            G.edges[(n,nbr)]['distance'] = 100. * (1. / min([G.degree(n), G.degree(nbr)]))

    # max dist
    max_distance = max([_ for u,v,_ in G.edges(data='distance')])
    for n, nbrs in G.adj.items():
        for nbr in nbrs:
            G.edges[(n,nbr)]['strength'] = 1 - (G.edges[(n,nbr)]['distance'] / max_distance)

    # add coloring by degree
    for n in G:
        degree = int(G.degree(n))
        size = len(G.node[n]['members'])
        G.node[n]['group']['degree'] = G.degree(n)
        G.node[n]['proportions']['degree'] = [dict(
            group=degree, row_count=size, value=size
        )]
    return G





def extract_matrices(G, index=None, **kwargs):
    # construct matrices
    #   A    => adjacency matrix
    #   M    => normalized node degree
    #   T    => normalized node degree
    if index is None:
        index = np.unique([
            __ for n in G for __ in G.node[n]['members']
            ])
    nTR = int(max(np.r_[len(index), np.ravel(index)+1]))
    A = nx.to_numpy_array(G)  # node x node
    M = np.zeros((nTR, A.shape[0]))    #   TR x node
    T = np.zeros((nTR, nTR))

    # return empty arrays if graph is empty
    if not len(G):
        return A, M, T

    # mapping from 'cube0_cluster0' => 0
    node_to_index = {n:i for i,n in enumerate(G)}
    node_to_members = dict(nx.get_node_attributes(G, 'members'))

    # loop over TRs to fill in C_rc, C_tp
    for TR in range(nTR):
        # find nodes containing TR 
        TR_nodes = [n for n in G if TR in G.node[n]['members']]
        #node_degrees = dict(G.degree(TR_nodes)).values()

        # find TRs for each edge sharing node
        node_index = [node_to_index[_] for _ in TR_nodes] 
        M[TR, node_index] += 1.0
        continue
        """
        source_TRs = [node_to_members[n] for n in TR_nodes]
        target_TRs = [node_to_members[nbr] for n in TR_nodes for nbr in G.neighbors(n)]
        #similar_TRs = list(set(__ for _ in source_TRs+target_TRs for __ in _))
        
        # normalized node degrees 
        #M[TR, node_index] += 1.0
        #T[TR, similar_TRs] += 1.0

        # count TRs multiple times
        similar_TRs = list(__ for _ in source_TRs+target_TRs for __ in _)
        TRs_counted = [similar_TRs.count(_) for _ in sorted(set(similar_TRs))]
        similar_TRs = sorted(set(similar_TRs))

        # normalized node degrees 
        M[TR, node_index] += 1.0
        T[TR, similar_TRs] += TRs_counted
        """
    # normalize
    T = M.dot(M.T)
    T /= T.max() 
    
    # return 
    return A, M, T
