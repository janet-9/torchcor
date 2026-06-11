# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Make the source tree importable so autodoc can read the docstrings.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "torchcor"
author = "Bei Zhou"
copyright = "2026, Bei Zhou and the torchcor contributors"
release = "1.0.2"
version = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from the source
    "sphinx.ext.autosummary",   # summary tables
    "sphinx.ext.napoleon",      # NumPy / Google style docstring sections
    "sphinx.ext.viewcode",      # add [source] links
    "sphinx.ext.intersphinx",   # cross-link to python / numpy / torch docs
    "sphinx.ext.mathjax",       # render the LaTeX in the docstrings
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock any runtime dependency that is not importable on the docs builder
# (e.g. torch / pyvista on a minimal Read the Docs image).  Locally, where the
# full stack is installed, nothing is mocked and autodoc renders everything.
autodoc_mock_imports = []
for _mod in ("torch", "pyvista", "pynvml", "scipy", "matplotlib", "pandas",
             "wfdb", "seaborn", "sklearn", "pygmsh", "vtk"):
    try:
        __import__(_mod)
    except Exception:
        autodoc_mock_imports.append(_mod)

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "torchcor"
html_logo = "logo.png"
html_theme_options = {
    "sidebar_hide_name": True,
}
