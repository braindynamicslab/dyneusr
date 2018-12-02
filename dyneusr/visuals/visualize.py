"""
D3.js visualizations and helper functions.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json 

from pathlib import Path
import IPython
import matplotlib.pyplot as plt
import numpy as np




CUSTOM_CSS = """
<style>
  .container { width:80% !important; margin:auto }
  .output_scroll {height: 80px !important;}
</style>
"""

def format_IFrame(path):
	from IPython.display import IFrame
	return IFrame(src=path, width="100%", height=800, frameBorder=0)

def format_HTML(html):
	from IPython.display import HTML
	if os.path.exists(html.split('://')[-1]):
		html = open(html.split('://')[-1], 'r').read()
	return HTML(html)

def display_HTML(html, static=False):
	""" Display html inside a Jupyter notebook.

	Notes
	-----
	Thanks to https://github.com/smartinsightsfromdata for the issue:
	https://github.com/MLWave/kepler-mapper/issues/10
	"""
	from IPython.core.display import display
	css = format_HTML(CUSTOM_CSS)
	if static:
		html = format_HTML(html)
	else:
		html = format_IFrame(html)
	# display
	display(css)
	display(html)
	return html


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
	return 

def visualize_force(js, template='movie', path_html='index.html', path_csv=None, path_json=None, path_graphs='graphs', path_assets=None, reset=True, static=False, show=False, figure=None, PORT=8000, **kwargs):
	""" Create index.html, index.csv, graphs/*.json
	"""
	### Read template HTML
	file_template = 'index.html'
	if template is not None:
		file_template = file_template.replace('.html', '-{}.html'.format(template))
	path_template = Path(__file__).resolve().parents[0] / 'templates' / file_template
	with open(str(path_template), 'r') as f:
		html = f.read()

	### Define path to output HTML
	path_html = Path(path_html)
	url = 'http://localhost:{}/{}'.format(PORT, str(path_html))

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
		path_csv =  path_html.parents[0] / path_assets / path_html.name.replace('.html', '.csv')
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
			json_str = json.dumps(json_str)
			div = """\n\t<div id="json_graph" data-json='{json_graph}' style="display:none"></div>"""
			div = div.format(json_graph=json_str)	
			html = html.replace('</head>', div+'\n\n</head>')


	### Write HTML to file
	print('[Force Graph] {}'.format(str(url)))
	with open(str(path_html), 'w') as f:
		f.write(html)	

	### Show HTML
	if show:
		figure = figure or plt.figure()
		display_HTML(str(url), static=static)

	return str(url)
