"""Configuration file for the Sphinx documentation builder.

See: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from datetime import datetime
import os
from pathlib import Path
import sys
from sphinx.ext import apidoc

# -- Path setup --------------------------------------------------------------
# Add your library (rkstiff) to sys.path so autodoc can import it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
project = "rkstiff"
author = "Patrick Whalen"
copyright = f"{datetime.now().year}, {author}"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]
autosummary_generate = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

templates_path = ["_templates"]
html_static_path = ["_static"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

pygments_style = "sphinx"

# -- Warning & error control -------------------------------------------------
# Avoid noisy or duplicate warnings
suppress_warnings = [
    "autodoc.duplicate_object",
    "myst.xref_missing",
    "ref.ref",
    "index",
]

# Treat warnings as non-fatal for local builds
nitpicky = False

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_title = f"{project} v{release}"
html_short_title = project
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": False,
    "special-members": "__init__",
    "inherited-members": False,
    "show-inheritance": True,
    "no-index": True,  # prevents duplicate object warnings from API files
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True

# -- Napoleon (docstring style) ----------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx links -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Autosummary -------------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- MathJax -----------------------------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- MyST Markdown -----------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Linkcheck ---------------------------------------------------------------
linkcheck_timeout = 10
linkcheck_ignore = [r"http://localhost"]


def run_apidoc_now():
    src_dir = Path(__file__).resolve().parents[2] / "rkstiff"
    out_dir = Path(__file__).resolve().parent / "api"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate .rst files
    apidoc.main(
        [
            "--force",
            "--separate",
            "--module-first",
            "--output-dir",
            str(out_dir),
            str(src_dir),
        ]
    )


run_apidoc_now()  # generate API docs immediately
