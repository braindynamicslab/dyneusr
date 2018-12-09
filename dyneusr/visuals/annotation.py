"""
Graph annotation helper functions.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx



NODE_HTML = """
<div id="tooltip-content">
  <div><b>Node ID:</b> {node_name}</div>
  <div><b>Node Size:</b> {node_size}</div>
  <div><b>Members:</b></div>
  <br>

  {member_tooltips}

</div>
"""

MEMBER_HTML = """
<span>{member_id} </span>
"""

IMG_HTML = """
<img class="tooltip-image" 
     src="{src}" 
     style="box-shadow: 0px 0px 0px 0px {color}">
"""


def annotate(G, verbose=False, **kwargs):
    """ Annotate G, format tooltips.
    
        #kwargs['image'][member_ids]
    """
    # node data = [(n, {...}), ...]
    node_data = {}
    if isinstance(G, nx.Graph):
        node_data = G.nodes(True)
    elif 'nodes' in G:
        node_data = [(d["id"], d) for d in G['nodes']]
    
    # unique ids
    member_uids = np.sort(np.unique([
        __ for n,d in node_data for __ in d['members']
        ]))

    # annotations
    annotation_map = dict(G.nodes(data=True))
    for ni,(n,d) in enumerate(node_data):
        
        # progress
        if verbose:
            print("Annotating node... NodeID:", n)
        member_ids = d['members']
        
        # member_html
        members_html = ""
        if kwargs.get('image') is not None:
            images = np.ravel(kwargs.get('image', []))
            if len(images) == 1:
                images = images[[0]]
            elif len(images) < len(member_uids):
                images = images[[ni]]
            else:
                images = images[member_ids]
            images = [IMG_HTML.format(src=_, color="#ffffff") for _ in images]
            members_html += "".join(images)
        else:
            members_htmls = [MEMBER_HTML.format(
                member_id=member_id,
                member_color=dict(node_data).get("color", "#777")
            ) for member_id in member_ids]  
            members_html += "".join(members_htmls)          

        # format tooltip for node
        node_tooltip = annotation_map[n].get('tooltip', '')
        if len(node_tooltip) < len(NODE_HTML):# is None: 
            node_tooltip = NODE_HTML.format(
                node_name=d['name'],
                node_size=len(member_ids), 
                member_tooltips=members_html
            )
        else:
            node_tooltip += """<div>{}</div>""".format(members_html)
    
        # annote node
        annotation_map[n] = dict(
            tooltip=node_tooltip, 
            image=members_html,
            )

    # nx
    if verbose:
        print("Annotating Graph...")
    nx.set_node_attributes(G, annotation_map)
    return G, annotation_map



def format_tooltips(G, **kwargs):
    """ Annotate G, format tooltips.
    """
    # node data = [(n, {...}), ...]
    node_data = {}
    if isinstance(G, nx.Graph):
        node_data = G.nodes(True)
    elif 'nodes' in G:
        node_data = [(d["id"], d) for d in G['nodes']]

    # unique ids
    member_uids = np.sort(np.unique([
        __ for n,d in node_data for __ in d['members']
        ]))

    # array of tooltips
    node_tooltips = []
    for n,d in node_data:

        # progress
        print("Formatting tooltip... NodeID:", n)
        member_ids = d['members']

        # member images
        images = d['image'][member_ids]
        images = [IMG_HTML.format(src=_) for _ in images]
    
        # format tooltip for node
        node_tooltip = NODE_HTML.format(
            node_id=n, node_name=d['name'],
            node_size=len(member_ids), 
            data_size=len(member_uids),
            images=images
        )
    
        # add to array
        node_tooltips.append(node_tooltip)
    
    # make numpy array
    return np.array(node_tooltips)
