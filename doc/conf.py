# -*- coding: utf-8 -*-

import sys
import os
import glob

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'scikits.bvp1lg'
copyright = u'2015, Pauli Virtanen'

import scikits.bvp1lg
version = scikits.bvp1lg.__version__
release = scikits.bvp1lg.__version__
exclude_patterns = ['_build']
pygments_style = 'sphinx'
default_role = "autolink"
html_static_path = ['_static']
html_style = 'fixup.css'


# -- Options for HTML output ----------------------------------------------

html_theme = 'nature'

# -- Options for extensions -----------------------------------------------

autosummary_generate = False
numpydoc_use_plots = True
plot_include_source = True
plot_html_show_formats = False
plot_template = """
{{ source_code }}

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.png
      {%- for option in options %}
      {{ option }}
      {% endfor %}

      {% if html_show_formats and multi_image -%}
        (
        {%- for fmt in img.formats -%}
        {%- if not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
        {%- endfor -%}
        )
      {%- endif -%}

      {{ caption }}
   {% endfor %}

"""

import math
phi = (math.sqrt(5) + 1)/2

font_size = 13*72/96.0  # 13 px

plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}
