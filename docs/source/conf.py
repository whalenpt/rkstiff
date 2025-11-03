"""Configuration file for the Sphinx documentation builder.

See: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from importlib.metadata import version as pkg_version, PackageNotFoundError
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

try:
    release = pkg_version("rkstiff")
except PackageNotFoundError:
    # Fallback to reading from the generated version file (if docs built locally)
    from pathlib import Path
    version_file = Path(__file__).resolve().parents[2] / "rkstiff" / "__version__.py"
    if version_file.exists():
        ns = {}
        exec(version_file.read_text(), ns)
        release = ns.get("__version__", "0.0.0")
    else:
        release = "0.0.0"

version = release.split("+")[0]  # strip local metadata if present

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

templates_path = []

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
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "navigation_depth": 3,
    "titles_only": False,
}

html_title = f"{project} v{release}"
html_short_title = project
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

html_js_files = [
    "custom.js",
]

html_static_path = ["_static"]

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": False,
    "special-members": "__init__",
    "inherited-members": True,
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
# Allow line breaks in equations and keep them centered
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "tags": "ams",
    },
    "chtml": {"scale": 1.0},
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


# -- Hide built-in Exception members from rendered HTML ----------------------
def skip_inherited_exception_members(app, what, name, obj, skip, options):
    """Skip built-in Exception attributes from HTML autodoc output."""
    if what == "exception" and name in {
        "add_note",
        "with_traceback",
        "args",
        "__init__",
        "__cause__",
        "__context__",
        "__suppress_context__",
        "__traceback__",
    }:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_inherited_exception_members)
