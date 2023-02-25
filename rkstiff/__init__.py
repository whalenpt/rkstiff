try:
    import importlib_metadata as metadata
except ModuleNotFoundError:
    import importlib.metadata as metadata

__version__ = metadata.version("rkstiff")
