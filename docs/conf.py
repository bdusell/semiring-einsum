# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Semiring Einsum'
copyright = '2019-2022, Brian DuSell'
author = 'Brian DuSell'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    'description' : 'Extensible PyTorch implementation of einsum that supports multiple semirings',
    'fixed_sidebar' : True,
    'logo' : 'logo.svg',
    'logo_name': True,
    'github_button': True,
    'github_repo' : 'semiring-einsum',
    'github_user' : 'bdusell'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# See https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'python' : ('https://docs.python.org/3', None),
    'numpy' : ('https://docs.scipy.org/doc/numpy', None),
    'pytorch' : ('https://pytorch.org/docs/stable', None)
}

html_favicon = 'logo.svg'

# For the sphinxcontrib-bibtex extension
bibtex_bibfiles = ['references.bib']

# For the ordering of autodoc-generated documentation.
autodoc_default_options = {
    'member-order' : 'bysource',
    'members' : True,
    'imported-members' : True
}
autodoc_class_signature = 'separated'
