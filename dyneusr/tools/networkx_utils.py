"""
Network plotting helper functions.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from collections import Counter

import matplotlib as mpl
mpl.use('TkAgg', warn=False)
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import networkx as nx



###############################################################################
###  Utils for working with networkx Graphs
###############################################################################
def get_layout_pos(G, layout="spring", layout_kws={}, pos=None, lens=None, **kwargs):
    if pos is None and layout is None:
        return None

    if pos is "inverse" and lens is not None:
        pos = np.copy(lens)
        pos = {n: list(pos[_].mean(0)) for n,_ in G.nodes.data("members")}    

    if layout:
        if "spring" in str(layout) and pos is "inverse":
            layout_kws = dict(dict(iterations=1), **layout_kws)
        layout = layout if callable(layout) else getattr(nx, layout+'_layout')
        try:
            pos = layout(G, pos=pos, **layout_kws)
        except Exception as e:
            print(e)
            pos = layout(G,  **layout_kws)
    if not isinstance(pos, dict):
        pos = {n:_ for n,_ in zip(G, pos)}
    pos = {n:list(pos[n]) for n in pos}
    nx.set_node_attributes(G, pos, "pos")
    return pos



def format_networkx(graph, meta=None, cmap="nipy_spectral_r", norm=None,  **kwargs):
    """ Format graph into networkx Graph object.
    """
    # Extract node, link data (require, for now)
    node_data = dict(graph['nodes'])
    link_dict = dict(graph['links'])
    
    # check meta data
    if meta is None:
        meta = sorted(set(__ for _ in node_data.values() for __ in _))
        
    # Construct Graph
    G = nx.Graph(link_dict)
    
    # color,norm
    cmap = cmap if callable(cmap) else plt.get_cmap(cmap)
    norm = norm if norm else mpl.colors.Normalize(np.min(meta)-1, np.max(meta)+1)
    to_hex = mpl.colors.to_hex
    
    # some lambdas
    _group_counter = lambda _: Counter(meta[list(_)])
    
    # Node Attributes
    nx.set_node_attributes(G, dict(node_data), "members")
    nx.set_node_attributes(G, {n: {
        "group": max(_group_counter(_)),
        "proportions": [{
            "group": g,
            "value": v,
            "row_count": len(_),
            "proportion": 100 * float(v / float(len(_))),
            } for g,v in _group_counter(_).items()],
        "row_count": len(_),
    } for n,_ in G.nodes("members")})


    # more lambdas
    _members = lambda _: set(G.nodes[_]["members"])
    _edge_members = lambda _: sorted(_members(_[0]) & _members(_[1]))
    
    # Edge Attributes
    nx.set_edge_attributes(G, {e: {
        "members": _edge_members(e),
        "group": max(_group_counter(_edge_members(e))),
    } for e in G.edges})
    
    
    # For visualization
    nx.set_node_attributes(G, {n: {
        "color_by": norm(G.nodes[n]["group"]),
        "color": to_hex(cmap(norm(G.nodes[n]["group"]))),
        "size": len(G.nodes[n]["members"]),
    } for n in G})
    nx.set_edge_attributes(G, {e: {
        "color_by": norm(G.edges[e]["group"]),
        "color": to_hex(cmap(norm((G.edges[e]["group"])))),
        "size": len(G.edges[e]["members"]),
        "weight": len(G.edges[e]["members"]),
    } for e in G.edges})
    
    # pos
    pos = get_layout_pos(G, **kwargs)
    nx.set_node_attributes(G, pos, "pos")

            
    # Return 
    return G



def draw_networkx(graph, ax=None, fig=None, nodes=True, edges=True, **kwargs):
    """ Draw the graph as a networkx graph
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    if ax is None:
        fig,ax = plt.subplots(1, 1, figsize=(5,5)) 
    else:
        fig = ax.get_figure()
    
         
    # Determine a fine size for nodes
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    area = width * height * fig.dpi
    
    
    # format nx
    if isinstance(graph, nx.Graph):
        G = graph.copy()
    else:
        G = format_networkx(dict(graph), **kwargs)

    # pos
    pos = nx.get_node_attributes(G, "pos")
    if pos is None or not len(list(pos)):
        pos = get_layout_pos(G, **kwargs)
    kwargs.update(pos = pos)

    
    # draw edges
    if edges is True:
        edge_zorder = kwargs.pop("edge_zorder", kwargs.pop("zorder", None))
        if kwargs.get("width") is None:
            edge_w = [np.sqrt(_) for u,v,_ in G.edges.data("size")]
            kwargs.update(width=edge_w)

        if kwargs.get("edge_color") is None:
            edge_c = [_ for u,v,_ in G.edges.data("color")]
            if not any(edge_c):
                edge_c = ['grey' for _ in G.edges]
            kwargs.update(edge_color=edge_c)

        # draw
        edges = nx.draw_networkx_edges(G, **kwargs, ax=ax)
        if edge_zorder is not None:
            edges.set_zorder(edge_zorder)

    
    # draw nodes
    if nodes is True:
        node_s0 = 0.5 * np.pi * area / len(G.nodes)
        node_r = np.sqrt(node_s0 / np.pi)
        node_edge = node_r / 3
        node_edge = kwargs.pop("node_edge", node_edge)
        node_edge_color = kwargs.pop("node_edge_color", "k") 
        node_zorder = kwargs.pop("node_zorder", kwargs.pop("zorder", None))

        if kwargs.get("node_size") is None:
            node_s = [node_s0 * np.sqrt(_) for n,_ in G.nodes.data("size")]
            kwargs.update(node_size=node_s)

        if kwargs.get("node_color") is None:
            node_c = [_ for n,_ in G.nodes.data("color")]
            if not any(node_c):
                node_c = [_ for n,_ in G.nodes.data("group")]
            kwargs.update(node_color=node_c)
        
        # draw
        nodes = nx.draw_networkx_nodes(G, **kwargs, ax=ax)
        if node_zorder is not None:
            nodes.set_zorder(node_zorder)
        if node_edge > 0:
            nodes.set_edgecolor(node_edge_color)
            nodes.set_linewidth(node_edge)
    
    # finish
    ax = despine(ax, **kwargs)
    return nodes, edges

def despine(ax, spines=['top','right'], **kwargs):
    if kwargs.get('keep_ticks') is not True:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in spines:
        ax.spines[spine].set_color('none')
    return ax



###############################################################################
### aliases
###############################################################################
format_nx = format_networkx
draw_nx = draw_networkx



###############################################################################
### Visualizing stages of MAPPER
###############################################################################
# TODO: move to mapper utils
def get_cover_cubes(lens=None, graph=None, cover=None, scale=False, **kwargs):
    # define bins
    cover = cover
    ids = np.arange(len(lens)).reshape(-1, 1)
    ilens = np.c_[ids, lens]
    bins = [tuple(_) for _ in cover.define_bins(ilens)]
    
    chunk_dist = cover.chunk_dist
    overlap_dist = cover.overlap_dist
    d = cover.d
    
    # plot for testing
    cover_cubes = {}
    for i, cube in enumerate(bins):
        # Compute bounds
        lower = d + (cube * chunk_dist)
        upper = lower + chunk_dist + overlap_dist
        cover_cubes[tuple(cube)] = np.r_['0,2', lower, upper]

    if scale:
        try:
            scaler = eval(graph['meta_data']['scaler'])
        except:
            scaler = MinMaxScaler()
        # scale
        stacked = np.vstack(cover_cubes.values())
        scaler.fit(stacked)
        for cube in cover_cubes:
            _ = scaler.transform(cover_cubes[cube])
            cover_cubes[cube] = _

    # cubes
    cover_cubes = cover_cubes
    
    # reindex by node
    coords = {n: tuple(graph['meta_nodes'][n]["coordinates"]) for n in graph["nodes"]}
    cover_cubes = {n: cover_cubes[coords[n]] for n in graph["nodes"]}
    return cover_cubes


def draw_cover(ax=None, cover_cubes=None, draw_all=False, max_draw=2, **kwargs):
    """ Add cover to scatter plot of projection/lens.
    """
    if ax is None:
        ax = plt.gca()

    # save for later
    xlim = np.ravel(ax.get_xlim())
    ylim = np.ravel(ax.get_ylim())
  

    if cover_cubes is None:
        cover_cubes = get_cover_cubes(**kwargs)

    # plot
    cmaps = [#lambda _: "cyan", lambda _: "deeppink"]
            #plt.get_cmap("Purples_r"), plt.get_cmap("Purples_r")]
            plt.get_cmap("bone")]*2# forDark2"), plt.get_cmap("Dark2")]

    axspan_funcs = [ax.axvspan, ax.axhspan]
    axline_funcs = [ax.axvline, ax.axhline]

    xbins,ybins = np.vstack([_ for cube,_ in cover_cubes.items()]).T
    xbins = np.ravel(sorted(set(xbins)))
    ybins = np.ravel(sorted(set(ybins)))
    hypercubes = np.c_[xbins, ybins]

    max_draw = 2
    if draw_all is True:
        max_draw = len(hypercubes)
    norm = mpl.colors.Normalize(0, 1)
    #norm = mpl.colors.Normalize(-0.3*max_draw, 0.3*max_draw)
    #norm = mpl.colors.Normalize(0, len(hypercubes)-2)#max_draw)
    d = hypercubes[1,:] - hypercubes[0,:]
    hypercubes = np.vstack([hypercubes, hypercubes[-1:, :] + d])
    for i, hypercube in enumerate(hypercubes[:]):
        for di, (axspan, axline) in enumerate(zip(axspan_funcs,axline_funcs)):
            c = cmaps[di](norm(0.4+0.3*int(i%2>0)))
            alpha = 1.0
            zo = i + 1
            if max_draw == 2 and (i < 2 or i > 3):
                alpha=0.2 
                zo = 0
            axspan(hypercubes[i,di], hypercubes[i,di]+d[di], alpha=0.1*alpha, fc=c, zorder=zo)
            axline(hypercubes[i,di], alpha=alpha, c=c, zorder=zo**2)
            axline(hypercubes[i,di]+d[di], alpha=alpha, c=c, zorder=zo**2+zo)
    


    # finish
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

    

def visualize_mapper_stages(data, lens, cover, graph, dG, **kwargs):
    """ Visualize stages of MAPPER.

    TODO
    -----
    - this needs to be tested...
    - move to mapper utils

    """

    #### Setup
    try:
        y = data.y.copy()
    except:
        y = data.target.copy()

    G = dG.G_.copy()
    
    # member_color 
    point_color = data.cmap(data.norm(y))

    # node color, size
    node_size = kwargs.get('node_size')
    if node_size is None:
        node_scale_by = kwargs.get('node_scale_by', 20)
        node_size = [node_scale_by*len(y[_]) for n,_ in G.nodes(data='members')]
    #node_color = [Counter(y[_]).most_common()[0][0] for n,_ in G.nodes(data='members')]
    node_color = [np.mean(y[_]) for n,_ in dG.G_.nodes(data='members')]
    node_color = data.cmap(data.norm(node_color))

    # edge color, size
    edge_size = kwargs.get('edge_size')
    if edge_size is None:
        edge_scale_by = kwargs.get('edge_scale_by', 1)
        edge_size = [edge_scale_by*_ for u,v,_ in G.edges(data='size')]
    edge_color = kwargs.get('edge_color')
    if edge_color is None:
        edge_sources = [G.nodes[u]['members'] for u,v in G.edges()]
        edge_targets = [G.nodes[v]['members'] for u,v in G.edges()]
        edge_color = [Counter(y[s + t]).most_common()[0][0] for s,t in zip(edge_sources, edge_targets)]
        edge_color = data.cmap(data.norm(edge_color))

    # init figure, subplots
    figsize = kwargs.get('figsize', (20,4))
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    
    #### Draw
    # 1. draw lens (axes: 1-3)
    for ax in axes[:3]:
        ax.scatter(*lens.T, c=point_color)

    # 2. draw cover (axes: 2)
    draw_cover(ax=axes[1], graph=graph, lens=lens, cover=cover)

    # 3. draw clusters (axes: 3)
    draw_networkx(G, lens=lens, pos="inverse", layout=None, 
            node_color=node_color, node_size=node_size, 
            edge_color=edge_color, width=edge_size, 
            alpha=0.5, edges=False, ax=axes[2])

    # 4. draw graph (axes: 4)
    draw_networkx(G, lens=lens, pos="inverse", layout=None, 
                node_color=node_color, node_size=node_size, 
                edge_color=edge_color, width=edge_size, 
                alpha=1.0, ax=axes[3])
    axes[3].axis('off')


    #### Finish
    for _ in axes:
        despine(_)

    return fig, axes




