"""Configuration file for the Sphinx documentation builder.

See: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from datetime import datetime
from pathlib import Path
import sys


# -- Suppress harmless duplicate-object warnings from autodoc/autosummary ----
import sphinx.util.logging

_OrigLog = sphinx.util.logging.SphinxLoggerAdapter.warning


def _sphinx_warning_filter(self, msg, *args, **kwargs):
    if isinstance(msg, str) and "duplicate object description of" in msg:
        return None  # ignore only this kind of warning
    return _OrigLog(self, msg, *args, **kwargs)


sphinx.util.logging.SphinxLoggerAdapter.warning = _sphinx_warning_filter
# ---------------------------------------------------------------------------


# -- Path setup --------------------------------------------------------------
# Add your library (rkstiff) to sys.path so autodoc can import it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
project = "rkstiff"
author = "Patrick Whalen"
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

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

templates_path = ["_templates"]
html_static_path = []  # Remove _static since it doesn't exist yet

exclude_patterns = [
    "_build",
    "build",
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
    "autosummary.stub",
    "autosummary.import_cycle",
]

# Treat warnings as non-fatal for local builds
nitpicky = False

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 2,  # Reduced from 4
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
autosummary_ignore_module_all = False

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
