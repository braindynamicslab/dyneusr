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
            #print(e)
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

    try:
        # new Cover API from kmapper==1.2.0
        bins = np.copy(cover.centers_)
    
        # transform each node
        cover_cubes = {}
        for i, center in enumerate(cover.centers_):
            lower = center - cover.radius_
            upper = center + cover.radius_
            cover_cubes[tuple(center)] = np.r_['0,2', lower, upper]

    except Exception as e:

        # support deprecated Cover API from kmapper==1.1.6
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

        # reindex by node
        coords = {n: tuple(graph['meta_nodes'][n]["coordinates"]) for n in graph["nodes"]}
        cover_cubes = {n: cover_cubes[coords[n]] for n in graph["nodes"]}

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

    return cover_cubes


def draw_cover(ax=None, cover_cubes=None, draw_all=True, max_draw=2, **kwargs):
    """ Add cover to scatter plot of projection/lens.
    """
    if ax is None:
        ax = plt.gca()

    # save for later
    xlim = np.ravel(ax.get_xlim())
    ylim = np.ravel(ax.get_ylim())
  

    if cover_cubes is None:
        cover_cubes = get_cover_cubes(**kwargs)

    # extract bins (TODO: probably a better way to do this)
    bins = np.vstack([_ for cube,_ in cover_cubes.items()])
    if len(bins.T) < 2:
        ybins = np.ravel(sorted(set(bins[:, 0])))
        bins = ybins.reshape(1, -1)
    else:
        # assume 2D
        xbins = np.ravel(sorted(set(bins[:, 0])))
        ybins = np.ravel(sorted(set(bins[:, 1])))
        bins = np.asarray([xbins, ybins])

    # save as hypercubes
    hypercubes = np.copy(bins)

    # draw
    if draw_all is True:
        max_draw = len(hypercubes)
    
    # cmap, norm for each dimension
    cmaps = [plt.get_cmap("jet")] * 2
    axspan_funcs = [ax.axhspan, ax.axvspan]
    axline_funcs = [ax.axhline, ax.axvline]
    norm = mpl.colors.Normalize(
        np.min([np.min(_) for _ in hypercubes]), 
        np.max([np.max(_) for _ in hypercubes])
        ) 

    # loop over dimensions
    for di, (axspan, axline) in enumerate(zip(axspan_funcs,axline_funcs)):
        if di >= len(hypercubes):
            continue

        # draw bounds of each cube along this dimension
        for i, hypercube in enumerate(hypercubes[di]):
            if i+1 >= len(hypercubes[di]):
                continue
            c = cmaps[di](norm(hypercubes[di][i]))
            alpha = 0.25 + (.5 * int((i+1)%2==0))
            zo = i + 1
            axspan(hypercubes[di][i], hypercubes[di][i+1], alpha=0.25*alpha, fc=c, zorder=zo)
            axline(hypercubes[di][i], alpha=alpha, c=c, zorder=zo**2)
            axline(hypercubes[di][i+1], alpha=alpha, c=c, zorder=zo**2+zo)
    
    # finish
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

    

def visualize_mapper_stages(data, y=None, lens=None, cover=None, graph=None, dG=None, **kwargs):
    """ Visualize stages of MAPPER.

    TODO
    -----
    - this needs to be tested...
    - move to mapper utils

    """

    #### Setup
    if y is None and hasattr(data, "y"):
        y = data.y.copy()
    elif y is None and hasattr(data, "target"):
        y = data.target.copy()

    try:
        G = dG.G_.copy()
    except:
        from .graph_utils import process_graph
        G = process_graph(graph, meta=y)

    
    # member color cmap
    cmap = kwargs.get('cmap') 
    if hasattr(data, "cmap"):
        cmap = data.cmap
    else:
        cmap = "nipy_spectral_r"
    cmap = cmap if callable(cmap) else plt.get_cmap(cmap)

    # member color norm
    norm = kwargs.get('norm') 
    if hasattr(data, "norm"):
        norm = data.norm
    else:
        norm = mpl.colors.Normalize(y.min(), y.max())
    
    # member_color 
    c = cmap(norm(y))
    c_hex = np.array([mpl.colors.to_hex(_) for _ in c])

    # node color, size
    node_size = kwargs.get('node_size')
    if node_size is None:
        node_scale_by = kwargs.get('node_scale_by', 5000)
        node_size = [node_scale_by*(len(_) / len(y)) for n,_ in G.nodes(data='members')]
    node_color = [Counter(c_hex[_]).most_common()[0][0] for n,_ in G.nodes(data='members')]


    # edge color, size
    edge_size = kwargs.get('edge_size')
    if edge_size is None:
        edge_scale_by = kwargs.get('edge_scale_by', 0.5)
        edge_size = [edge_scale_by*_ for u,v,_ in G.edges(data='size')]
    edge_color = kwargs.get('edge_color')
    if edge_color is None:
        edge_sources = [G.nodes[u]['members'] for u,v in G.edges()]
        edge_targets = [G.nodes[v]['members'] for u,v in G.edges()]
        edge_color = [Counter(c_hex[s + t]).most_common()[0][0] for s,t in zip(edge_sources, edge_targets)]


    # layout (for last stage only)
    layout = kwargs.get('layout', None)
    pos = kwargs.get('pos', None)
    if layout is None:
        pos = "inverse"

    # init figure, subplots
    figsize = kwargs.get('figsize', (20,4))
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    
    #### Draw
    # ensure the lens is 2D
    lens2D = lens.copy()
    if len(lens.T) < 2:
        lens2D = np.c_[np.zeros_like(lens) + lens.mean(), lens] 
    elif len(lens.T) > 2:
        lens2D = lens2D[:, :2]

    # 1. draw lens (axes: 1-3)
    for ax in axes[:3]:
        ax.scatter(*lens2D.T, c=c, s=np.max(node_size)*.05)
        
        # adjust xlim if 1D
        if len(lens.T) < 2:
            ax.set_xlim(lens.min(), lens.max())

    # 2. draw cover (axes: 2)
    draw_cover(ax=axes[1], graph=graph, lens=lens2D, cover=cover)

    # 3. draw clusters (axes: 3)
    draw_networkx(G, lens=lens2D, pos="inverse", layout=None, 
            node_color=node_color, node_size=node_size, 
            edge_color=edge_color, width=edge_size, 
            alpha=0.5, edges=False, ax=axes[2])

    # 4. draw graph (axes: 4)
    draw_networkx(G, lens=lens2D, pos=pos, layout=layout, 
                node_color=node_color, node_size=node_size, 
                edge_color=edge_color, width=edge_size, 
                alpha=1.0, ax=axes[3])
    if layout is None:
        axes[3].set_xlim(axes[2].get_xlim())
        axes[3].set_ylim(axes[2].get_ylim())
    axes[3].axis('off')


    #### Finish
    for ax in axes:

        # despine, based on number of dimensions
        if len(lens.T) > 1:
            despine(ax, spines=['top','right'])
        else:
            despine(ax, spines=['top', 'right', 'bottom', 'left'])

        # tight layout
        ax.set_aspect('equal')
        fig.tight_layout(w_pad=2.0)

    return fig, axes




