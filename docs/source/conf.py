# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

project = "Genetic Engine"
copyright = "2022, Alcides Fonseca et al."
author = "Alcides Fonseca et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.spelling",
    "sphinx_autodoc_typehints",
]

autoapi_type = "python"
autoapi_dirs = ["../../geneticengine"]
autoapi_python_class_content = "both"
autoapi_add_objects_to_toctree = False

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 3

autosectionlabel_prefix_document = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
