"""rkstiff package initialization."""

import logging
from .__version__ import version as __version__

# Prevent "No handlers could be found" warnings when rkstiff is imported by
# applications that have not configured logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
