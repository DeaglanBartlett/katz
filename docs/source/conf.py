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
import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../katz/"))
sys.path.insert(0, os.path.abspath("../../scripts/"))

autodoc_mock_imports = ["mpi4py", "astropy", "esr"]

# The tutorial's examples are doctests (make doctest). They run from the repository root,
# where data/ is, and the part that scores an ESR library is skipped without ESR and mpi4py
doctest_global_setup = f"""
import os
os.chdir({os.path.abspath("../..")!r})
try:
    import esr.generation.duplicate_checker
    import mpi4py.MPI
    have_esr = True
except ImportError:
    have_esr = False
"""
# The classes are documented in their __init__ docstrings, so show these with the class
autoclass_content = "both"


# -- Project information -----------------------------------------------------

project = "Katz"
copyright = "2023-2026, Deaglan J. Bartlett and Harry Desmond"
author = "Deaglan J. Bartlett and Harry Desmond"

# The full version, including alpha/beta/rc tags, read from the package as in setup.py
with open(os.path.abspath("../../katz/prior.py")) as f:
    release = re.search(r'^__version__ = "(.+)"', f.read(), re.MULTILINE).group(1)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.imgconverter",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
