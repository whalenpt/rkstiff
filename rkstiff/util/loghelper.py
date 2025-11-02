"""
Logging helper utilities for rkstiff solvers.

Provides functions for parsing log levels, setting up loggers, and configuring logging for solver classes.
All functions use NumPy docstring style and are compatible with Sphinx autodoc.
"""

import logging


def _parse_loglevel(loglevel):
    """
    Convert loglevel input to logging integer constant.

    Parameters
    ----------
    loglevel : str or int
        Logging level as string (e.g., 'INFO') or integer constant.

    Returns
    -------
    int
        Numeric logging level.

    Raises
    ------
    ValueError
        If loglevel string is invalid.
    """
    if isinstance(loglevel, str):
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        return numeric_level
    return loglevel


def _get_level_name(level):
    """
    Get the string name for a logging level.

    Parameters
    ----------
    level : int
        Numeric logging level.

    Returns
    -------
    str
        Level name (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    level_names = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARNING',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRITICAL'
    }
    return level_names.get(level, 'Level %s' % level)


def setup_logger(name, loglevel='WARNING'):
    """
    Set up a logger with the specified name and level.

    Parameters
    ----------
    name : str
        Name for the logger (typically __name__ or class name).
    loglevel : str or int, optional
        Logging level. Can be:
        - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        - Integer: logging.DEBUG, logging.INFO, etc.
        - Default: 'WARNING'
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Set the logging level using helper
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)
    
    # Add handler if logger doesn't have one (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def set_log_level(logger, loglevel):
    """
    Set the logging level for an existing logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to configure.
    loglevel : str or int
        Logging level as string or integer constant.

    Raises
    ------
    ValueError
        If loglevel string is invalid.
    """
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)


def get_solver_logger(solver_class, loglevel='WARNING'):
    """
    Create a logger for a solver instance.

    Parameters
    ----------
    solver_class : type
        The solver class (e.g., ETD35, IF34).
    loglevel : str or int, optional
        Initial logging level.

    Returns
    -------
    logger : logging.Logger
        Configured logger for the solver.
    """
    logger_name = 'rkstiff.%s' % solver_class.__name__
    return setup_logger(logger_name, loglevel)