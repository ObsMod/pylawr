# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import matplotlib as mpl
import os
import sys
import re
from datetime import datetime
import pylawr
mpl.use('agg')                      # Without this sphinx_gallery has problems

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))


# -- Project information -----------------------------------------------------

project = 'pylawr'
copyright = datetime.now().strftime('%Y, '
                                    'Meteorological Institute, '
                                    'Universität Hamburg')
author = ('pylawr developers')

# The full version, including alpha/beta/rc tags
version = re.match('\d+\.\d+', pylawr.__version__).group()
release = pylawr.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosectionlabel',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',
     'gallery_dirs': 'examples',
     'filename_pattern': '/*.py',
     'doc_module': ('pylawr'),
}

# Define bibtex file for sphinxcontrib.bibtex
bibtex_bibfiles = ['appendix/references.bib', 'appendix/publications.bib']
bibtex_encoding = 'latin'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autosummary_generate = True

# -- pybtex definitions for changing citation reference labels ----------------
# from Wradlib doc config
from pybtex.style.formatting.alpha import Style  # noqa
from pybtex.style.labels.alpha import LabelStyle, _strip_nonalnum  # noqa
from pybtex.plugin import register_plugin  # noqa


# citation style from wradlib
class CitationLabelStyle(LabelStyle):
    def format_label(self, entry):
        if entry.type == "book" or entry.type == "inbook":
            label = self.author_editor_key_label(entry)
        elif entry.type == "proceedings":
            label = self.editor_key_organization_label(entry)
        elif entry.type == "manual":
            label = self.author_key_organization_label(entry)
        else:
            label = self.author_key_label(entry)
        if "year" in entry.fields:
            return '{0:s}, {1:s}'.format(label, str(entry.fields["year"]))
        else:
            return label

    def format_lab_names(self, persons):
        numnames = len(persons)
        person = persons[0]
        result = _strip_nonalnum(
            person.prelast_names + person.last_names)
        if numnames > 1:
            result += " et al."
        return result


class CitationStyle(Style):
    default_label_style = 'cit'


register_plugin('pybtex.style.labels', 'cit', CitationLabelStyle)
register_plugin('pybtex.style.formatting', 'citstyle', CitationStyle)

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'wradlib': ('http://docs.wradlib.org/en/latest/', None),
    'cartopy': ('http://scitools.org.uk/cartopy/docs/latest/', None),
    'h5py' : ('https://docs.h5py.org/en/stable/', None),
}
