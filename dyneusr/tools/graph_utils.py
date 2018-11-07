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

from collections import defaultdict, Counter

import multiprocessing as mp

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats

from sklearn.preprocessing import Normalizer, LabelEncoder
import networkx as nx


def process_meta(meta_, labels=None):
 
    # process each column of meta
    meta_sets = dict()
    meta_labels = dict()
    for i,meta_col in enumerate(meta_.columns):
        print("Processing meta column: {}".format(meta_col))
        meta = np.ravel(meta_[meta_col].values.copy())
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
            meta_label = ['Group '+str(_+1) for _ in meta_label]

        
        elif len(set(meta)) > 9:
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
            meta_str = {False: '\u03BC (= {:0.2f})'.format(np.mean(yi_nz)),
                        True: '\u03BC + {:1.0f}\u03C3'}
            meta_label = [meta_str[np.abs(float(_)) > 0].format(float(_)) for _ in meta_label if not np.isnan(_)]
            meta_label = ['n/a'] + meta_label

    
        # labels for legend
        meta_sets[meta_col] = [_ for _ in np.sort(np.unique(meta))]
        if meta_label is not None:
            meta_labels[meta_col] = [_ for _ in meta_label]
        print("  [+] found {} unique groups.".format(len(meta_sets[meta_col])))

        # re-assign meta
        meta_[meta_col] = meta.copy()

    return meta_, meta_sets, meta_labels


def process_graph(graph=None, meta=None, tooltips=None, color_by=0, labels=None, **kwargs):
    # convert to nx
    nodelist = dict(graph['nodes'])
    edgelist = dict(graph['links'])
    memberlist = np.sort(np.unique([
        __ for _ in nodelist.values() for __ in _]))
    
    if meta is None:
        meta = pd.DataFrame().assign(
            data_id=np.arange(np.max(memberlist)+1).astype(str), 
            default=0,
            )
    elif not isinstance(meta, pd.DataFrame):
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
    # add some defaults
    meta = meta.assign(
        data_id=np.arange(len(meta)).astype(str),
        default='0'
        )

    # normalize meta
    # TODO: move all of this logic into color map utils
    #meta = Normalizer().fit_transform(meta.reshape(-1, 1))
    # bin meta (?)
    meta, meta_sets, meta_labels  = process_meta(meta, labels=labels)

    # index
    node_to_index = {n:i for i,n in enumerate(nodelist)}
    index_to_node = {i:n for i,n in enumerate(nodelist)}

    # tooltips (TODO: should this be here)
    if tooltips is None:
        tooltips = np.array(list(nodelist.keys()))
    tooltips = np.array(tooltips).astype(str)


    ### NODES
    G = nx.Graph()
    G.graph['label'] = meta_labels
    G.graph['groups'] = meta_sets
    G.graph['color_by'] = color_by if color_by in meta.columns else meta.columns[color_by]
    for node_id, (name, members) in enumerate(nodelist.items()):
        # define node_dict for G
        members = list(sorted(members))
        tooltip = tooltips[node_id]

        coloring = kwargs.get('rsn_color','separate')
        # proportions by column
        if coloring is 'separate': # Color networks separately or put all together in one graph
            group = dict()
            proportions = dict()
            for meta_col in meta.columns:
                groups = meta[meta_col].values[members] 
                proportions[meta_col] = [
                    dict(group=int(_), row_count=len(groups), value=int(c))
                    for _,c in Counter(groups).most_common() #metaset
                    ]
                group[meta_col] = int(Counter(groups).most_common()[0][0])

                # format node dict
                node_dict = dict(
                    id=int(node_id),
                    name=name,
                    tooltip=tooltip,
                    members=members,
                    proportions=proportions,
                    group=group,
                    # node color, size
                    size=len(members),
                ) 

        elif coloring is 'together':
            # multilabel
            proportions = dict()
            groups = meta.values[members, :-2] 
            allgroups = [np.nonzero(memlabels) for memlabels in groups]
            allgroups = [el for subarr in allgroups for subsubarr in subarr for el in subsubarr]   
            proportions['multi'] = [
                    dict(group=int(_), row_count=len(allgroups), value=int(c))
                    for _,c in Counter(allgroups).most_common() 
                    ]
            group = int(Counter(allgroups).most_common()[0][0])

            # format node dict
            node_dict = dict(
                id=int(node_id),
                name=name,
                tooltip=tooltip,
                members=members,
                proportions=proportions,
                group=group,
                # node color, size
                size=len(members),
            ) 

        
        # update G
        G.add_node(name, **node_dict)
        
    ### EDGES
    for (source, targets) in edgelist.items():
        # add edge for each target
        for target in targets:
            source_id = node_to_index[source]
            target_id = node_to_index[target]

            source_node = G.nodes[source]
            target_node = G.nodes[target]

            # find member intersection
            members_index = [i for i,_ in enumerate(target_node['members'])
                             if _ in source_node['members']]

            # define edge dict
            edge_dict = dict(
                value=len(members_index),
                size=len(members_index)
                # edge color, size
            )

            # update G
            G.add_edge(source, target, **edge_dict)

    # more attribues
    nx.set_node_attributes(G, {n: {
        'degree': G.degree[n],
        } for n in G})
    nx.set_edge_attributes(G, {e: {
        'distance': 100. * (1. / min([G.degree[n] for n in e])),
        'distance': (1. / min([G.degree[n] for n in e]))**2,

        } for e in G.edges})    
    # max dist
    max_distance = np.max(list(dict(nx.get_edge_attributes(G, 'distance')).values()))
    print(max_distance)
    nx.set_edge_attributes(G, {e: {
        'strength': 1 - (G.edges[e]['distance'] / max_distance),
        } for e in G.edges})         
    return G





def extract_matrices(G):
    # construct matrices
    #   A    => adjacency matrix
    #   M    => normalized node degree
    #   T    => normalized node degree
    data = np.sort(np.unique([
        __ for n,_ in G.nodes(data=True) for __ in _['members']
        ]))
    nTR = max(data.max()+1, data.shape[0])
    A = nx.to_numpy_matrix(G)          # node x node
    M = np.zeros((nTR, A.shape[0]))    #   TR x node
    T = np.zeros((nTR, nTR))

    # mapping from 'cube0_cluster0' => 0
    node_to_index = {n:i for i,n in enumerate(G.nodes)}
    node_to_members = dict(nx.get_node_attributes(G, 'members'))

    # loop over TRs to fill in C_rc, C_tp
    for TR in range(nTR):
        # find nodes containing TR 
        TR_nodes = [n for n,d in G.nodes(data=True) if TR in d['members']]
        #node_degrees = dict(G.degree(TR_nodes)).values()

        # find TRs for each edge sharing node
        node_index = [node_to_index[_] for _ in TR_nodes]  
        source_TRs = [node_to_members[n] for n in TR_nodes]
        target_TRs = [node_to_members[n] for (_,n) in G.edges(TR_nodes)]
        similar_TRs = list(set(__ for _ in source_TRs+target_TRs for __ in _))

        # normalized node degrees 
        M[TR, node_index] += 1.0
        T[TR, similar_TRs] += 1.0
       
    # return 
    return A, M, T
