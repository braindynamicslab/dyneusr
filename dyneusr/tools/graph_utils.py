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



def process_graph(graph=None, meta=None, tooltips=None, color_by=0, labels=None, **kwargs):
    # convert to nx
    nodelist = dict(graph['nodes'])
    edgelist = dict(graph['links'])
    memberlist = np.sort(np.unique([
        __ for _ in nodelist.values() for __ in _]))
    if meta is None:
        meta = np.zeros_like(memberlist)

    # normalize meta
    # TODO: move all of this logic into color map utils
    #meta = Normalizer().fit_transform(meta.reshape(-1, 1))
    # bin meta (?)
    if isinstance(meta, pd.DataFrame):
        if isinstance(color_by, str):
            meta = meta[color_by].values
        else:
            meta = meta.iloc[:, color_by].values
    elif np.ndim(meta) > 1:
        meta = meta[:, color_by]
    meta = np.ravel(meta).copy()
    meta_label = None
    if labels is not None:
        meta_label = list(labels)
    if len(set(meta)) > 9:
        # zscore
        yi = meta.copy()
        yi_nz = yi[~np.isnan(yi)]
        zi = stats.zscore(yi_nz)
        # set each zscore to the lower bound of absolute value
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
        #print(set(meta))
        #print(meta_bins)

        # map meta to labels
        meta_str = {False: '\u03BC (= {:0.2f})'.format(np.mean(yi_nz)),
                    True: '\u03BC + {:1.0f}\u03C3'}
        meta_label = [meta_str[np.abs(float(_)) > 0].format(float(_)) for _ in meta_label if not np.isnan(_)]
        meta_label = ['n/a'] + meta_label
        #print(meta_label)

        # now re-encode labels to meta
        #encoder = LabelEncoder().fit_transform(meta)
        #yi = encoder.fit_transform(meta)
        #yi_bins = np.linspace(yi.min(), yi.max(), num=min(5, len(set(yi))), endpoint=True)
        #meta = np.digitize(yi, yi_bins, right=True)
        #meta = yi_bins[meta]
    elif str(meta[0]).isalpha():
        encoders = defaultdict(LabelEncoder)
        yi = encoders[color_by].fit_transform(meta)
        yi_bins = np.linspace(yi.min(), yi.max(), num=min(5, len(set(yi))), endpoint=True)
        meta = np.digitize(yi, yi_bins, right=True)
        meta = yi_bins[meta]
        meta_label = [str(_) for _ in encoders[color_by].classes_]

    # labels for legend
    metaset = np.sort(np.unique(meta))
    print("Found {} unique groups.".format(len(metaset)))
    #print(metaset)
    #print(meta_label)


    # index
    node_to_index = {n:i for i,n in enumerate(nodelist)}
    index_to_node = {i:n for i,n in enumerate(nodelist)}

    # tooltips (TODO: should this be here)
    if tooltips is None:
        tooltips = np.array(list(nodelist.keys()))
    tooltips = np.array(tooltips).astype(str)


    ### NODES
    G = nx.Graph()
    G.graph['label'] = meta_label 
    G.graph['groups'] = list(metaset)
    for node_id, (name, members) in enumerate(nodelist.items()):
        # define node_dict for G
        members = list(sorted(members))
        groups = meta[members]
        proportions = [
            dict(group=int(_), row_count=len(groups), value=int(c))
            for _,c in Counter(groups).most_common() #metaset
            ]
        group = Counter(groups).most_common()[0][0]
        tooltip = tooltips[node_id]

        # format node dict
        node_dict = dict(
            id=int(node_id),
            name=name,
            tooltip=tooltip,
            members=members,
            proportions=proportions,
            group=int(group),
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
