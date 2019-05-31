"""
D3.js visualizations and helper functions.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json 
import subprocess
import functools 

from pathlib import Path

import IPython
import matplotlib.pyplot as plt
import numpy as np



def format_IFrame(path, **kwargs):
    defaults = dict(src=path, width="100%", height=800, frameBorder=0)
    return IPython.display.IFrame(**dict(defaults, **kwargs))


def format_HTML(html):
    loc, *src = html.split('://', 1)
    if loc in ['file', 'http', 'https']:
        src = src[0] if loc in 'file' else src[0].split('/', 1)[-1]
        with open(src, 'r') as fid:
            html = fid.read()
    return IPython.display.HTML(html)


CUSTOM_CSS = """
<style>
  .container { width:80% !important; margin:auto }
  .output_scroll {height: 80px !important;}
</style>
"""

def display_HTML(src="", figure=None, static=False):
    """ Display html inside a Jupyter notebook.

    Notes
    -----
    Thanks to https://github.com/smartinsightsfromdata for the issue:
    https://github.com/MLWave/kepler-mapper/issues/10
    """
    # check figure
    figure = figure or plt.figure()
    css = format_HTML(CUSTOM_CSS)
    if static:
        html = format_HTML(src)
    else:
        html = format_IFrame(src)
    # display
    IPython.display.display(css)
    IPython.display.display(html)
    return html

def in_notebook():
    """ Returns ``True`` if the module is running in IPython kernel,
      ``False`` if in IPython shell or other Python shell.
    """
    from IPython import get_ipython
    return get_ipython() is not None

def json_dump(obj, fp):
    """ Converts np.int64 to ints
    """
    def default(o):
        if isinstance(o, np.int64): 
            #print(o)
            return int(o)
        if isinstance(o, np.ndarray): 
            #print(o)
            return list(o)  
        raise TypeError

    json.dump(obj, fp, default=default)
    return None




def http_server(port=None, host='localhost'):
    """ Start http.server listening to port.

    Usage
    -----
    # open specific port
    status = open_port(8800)

    # open next available
    port = open_port(None)

    """     
    if port is not None and port < 0:
        # open http.server on port, or next available
        p = http_server(port=8000, host=host) 

        # check status, try next (until found)
        while p.status is False:
            p = http_server(port=p.port+1, host=host)
        return p

    else:
        # open http.server on port
        p = subprocess.Popen(
            'python3 -m http.server {} -b {} &'.format(port, host), 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        p.host = host
        p.port = port


        # check status, return process
        try:
            o, e = p.communicate(timeout=1)
            print("Already serving {}:{} ".format(host, port))
            p.status = False
        except subprocess.TimeoutExpired as e:
            print("Serving HTTP on {}:{} ...".format(host, port))
            p.status = True        
        return p
                


    
def visualize_force(js, template=None, path_html='index.html', path_csv=None, path_json=None, path_graphs='graphs', path_assets=None, reset=True, static=False, show=True, figure=None, port=8000, **kwargs):
    """ Create index.html, index.csv, graphs/*.json
    """
    ### check for deprecated options
    if kwargs.get('PORT'):
        # PORT => port
        port = kwargs.pop('PORT')

    ### no need to open a new port if static
    if static is True:
        port = None
        
    ### Run http.server on port
    HTTP = http_server(port=port, **kwargs)

    ### Read template HTML
    file_template = 'index.html'
    if template and template not in ('movie',): 
        file_template = file_template.replace('.html', '-{}.html'.format(template))
    path_template = Path(__file__).resolve().parents[0] / 'templates' / file_template
    with open(str(path_template), 'r') as f:
        html = f.read()

    ### Define path to output HTML
    path_html = Path(path_html)
    url = 'http://{}:{}/{}'.format(HTTP.host, HTTP.port, str(path_html))

    ### Path to save assests
    if path_assets is None:
        path_assets = '.'
    path_assets = Path(path_assets)
    if not path_assets.exists():
        os.makedirs(str(path_assets))

    ### Check graphs director
    graphs_dir = path_html.parents[0] / path_assets / path_graphs
    if not graphs_dir.exists():
        os.makedirs(str(graphs_dir))

    ### Write graphs/*.json
    if path_json is None:
        path_json = graphs_dir / path_html.name.replace('.html', '.json')
    path_json = Path(path_json)
    with open(str(path_json), 'w') as f:
        json_dump(js, f)
        

    ### Write index.csv
    if path_csv is None:
        #path_csv =  path_html.parents[0] / path_assets / path_html.name.replace('.html', '.csv')
        path_csv = graphs_dir / path_html.name.replace('.html', '.csv')
    # clean up paths
    path_csv = Path(path_csv)
    path_csv_rel = path_csv.relative_to(path_html.parents[0])
    path_json_rel = path_json.relative_to(path_html.parents[0])
    # update html with csv path
    html = html.replace('index.csv', str(path_csv_rel))
    if path_csv.exists() and not reset:
        with open(str(path_csv), 'a') as f:
            f.write('\n')
            f.write(str(path_json_rel))
    else:
        with open(str(path_csv), 'w') as f:
            f.write('json\n')
            f.write(str(path_json_rel))

    ### Load template HTML
    # TODO: should probably seperate these into seperate files
    if static is True:        
        ### Rename path with -static
        #path_html = Path(str(path_html).replace('.html', "-static.html"))
        url = "file:///" + str(path_html.resolve())

        ### Add js directly to HTML
        with open(str(path_json), 'r') as f:
            json_str = json.load(f)
            json_str = json.dumps(json_str).replace("\"", "\"")
            div = """\n\t<div id="json_graph" data-json='{json_graph}' style="display:none"></div>"""
            div = div.format(json_graph=json_str)    
            html = html.replace('</head>', div+'\n\n</head>')


    ### Write HTML to file
    print('[Force Graph] {}'.format(str(url)))
    with open(str(path_html), 'w') as f:
        f.write(html)    

    ### Show HTML
    if show and in_notebook():
        display_HTML(str(url), figure=figure, static=static)
    elif show is True:
        try:
            import webbrowser
            webbrowser.open(str(url))
        except ImportError as e:
            print(e)
            print('Hint: requires Python webbrowser module...')

    # cache some things
    HTTP.url = str(url)
    HTTP.src = str(path_html)
    HTTP.html = str(html)
    HTTP.json = dict(js)

    # define some helper functions (TODO: clean up HTTP handling, move to utils)
    HTTP.display = functools.partial(display_HTML, src=HTTP.url, static=static)
    try:
        import webbrowser
        HTTP.open = functools.partial(webbrowser.open, url=HTTP.url)
    except:
        HTTP.open = functools.partial(print, HTTP.url)

    return HTTP
