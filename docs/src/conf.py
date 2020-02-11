# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_material
import baikal

# -- Project information -----------------------------------------------------

project = "baikal"
copyright = "2019, Alejandro González Tineo"
author = "Alejandro González Tineo"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = baikal.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"
highlight_language = "python3"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- HTML theme settings ------------------------------------------------

html_show_sourcelink = True
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

extensions.append("sphinx_material")
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "baikal {}".format(version),
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXXXX',
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://baikal.github.io",
    # Set the color and the accent color
    "color_primary": "003366",
    # 'color_accent': 'cyan',
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/alegonz/baikal/",
    "repo_name": "baikal",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 1,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": True,
    "heroes": {
        "index": "A graph-based functional API for building complex scikit-learn pipelines.",
    },
}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

html_use_index = True
html_domain_indices = True


# -- Options for autodoc -----------------------------------------------------

autoclass_content = "class"
autodoc_typehints = "none"
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
}


# -- Options for autosummary -----------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = False


# -- Options for napoleon--- -----------------------------------------------------

napoleon_use_rtype = False


# -- -------------------------------------------------------------------------
def setup(app):
    app.add_stylesheet("custom.css")
