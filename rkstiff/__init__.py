try:
    import importlib_metadata as metadata
except:
    import importlib.metadata as metadata

__version__ = metadata.version("rkstiff")
