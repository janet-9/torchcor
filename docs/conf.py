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
    "sphinx_design",            # tabs, cards, grids, badges
    "sphinx_copybutton",       # copy button on code blocks
]

# sphinx-copybutton: strip prompts so "copy" yields runnable commands
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

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
    "source_repository": "https://github.com/sagebei/torchcor/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sagebei/torchcor",
            "html": '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# Small CSS polish (wider content, nicer code blocks).
html_static_path = ["_static"]
html_css_files = ["custom.css"]
