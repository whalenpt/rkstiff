r"""
Logging helper utilities
========================

Logging helper utilities for :mod:`rkstiff` solvers.

This module provides standardized logging configuration tools for
solver classes and submodules within the :mod:`rkstiff` framework.
It ensures consistent, hierarchical log naming (e.g., ``rkstiff.IF4``),
and human-readable output formatting suitable for both console use and
debugging during solver development.

Overview
--------

- :func:`_parse_loglevel` — Convert user-specified log level to a numeric constant.
- :func:`get_level_name` — Return string name for a numeric logging level.
- :func:`setup_logger` — Configure and return a new logger.
- :func:`set_log_level` — Adjust the level of an existing logger.
- :func:`get_solver_logger` — Create a standardized logger for solver classes.

Example
-------

.. code-block:: python

    from rkstiff.util.loghelper import get_solver_logger

    class IF4:
        def __init__(self):
            self.logger = get_solver_logger(self.__class__, "INFO")
            self.logger.info("Initialized IF4 solver")

    # Output:
    # 2025-11-02 10:30:45 - rkstiff.IF4 - INFO - Initialized IF4 solver
"""

import logging
from typing import Union


def _parse_loglevel(loglevel: Union[str, int]) -> int:
    r"""
    Convert a log level (string or numeric) into a Python ``logging`` integer constant.

    Parameters
    ----------
    loglevel : str or int
        Logging level, e.g. ``"INFO"`` or ``logging.DEBUG``.

    Returns
    -------
    int
        Numeric logging level constant.

    Raises
    ------
    ValueError
        If the string does not match a valid logging level name.

    Examples
    --------
    >>> _parse_loglevel("INFO")
    20
    >>> _parse_loglevel(logging.DEBUG)
    10
    """
    if isinstance(loglevel, str):
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")
        return numeric_level
    return loglevel


def get_level_name(level: int) -> str:
    r"""
    Return the canonical string name for a numeric logging level.

    Parameters
    ----------
    level : int
        Logging level constant (e.g. ``logging.INFO``).

    Returns
    -------
    str
        Corresponding level name, such as ``"DEBUG"``, ``"INFO"``, or ``"ERROR"``.

    Examples
    --------
    >>> get_level_name(logging.INFO)
    'INFO'
    >>> get_level_name(15)
    'Level 15'
    """
    level_names = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    return level_names.get(level, f"Level {level}")


def setup_logger(name: str, loglevel: Union[str, int] = "WARNING") -> logging.Logger:
    r"""
    Create and configure a logger with standardized formatting.

    The logger uses timestamped output of the form::

        YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - message

    If a logger with the given name already exists, this function will not
    add additional handlers (to avoid duplicated messages).

    Parameters
    ----------
    name : str
        Name for the logger (e.g. ``"rkstiff.solver.IF4"`` or ``__name__``).
    loglevel : str or int, optional
        Logging level as a string or integer. Default is ``"WARNING"``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    This helper ensures a consistent format across all :mod:`rkstiff` modules.
    Typically used via :func:`get_solver_logger` for class-based solvers.

    Examples
    --------
    >>> logger = setup_logger("rkstiff.IF4", "INFO")
    >>> logger.info("Solver initialized")
    2025-11-02 10:30:45 - rkstiff.IF4 - INFO - Solver initialized
    """
    logger = logging.getLogger(name)
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)

    # Only add a handler if none exist (avoids duplicate output)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_log_level(logger: logging.Logger, loglevel: Union[str, int]) -> None:
    r"""
    Set or update the logging level of an existing logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to modify.
    loglevel : str or int
        New logging level (e.g. ``"DEBUG"`` or ``logging.INFO``).

    Raises
    ------
    ValueError
        If the provided string does not correspond to a valid logging level.

    See Also
    --------
    setup_logger : Create and configure a new logger.
    get_solver_logger : Generate a standardized solver logger.

    Examples
    --------
    >>> logger = setup_logger("rkstiff.IF4", "WARNING")
    >>> set_log_level(logger, "DEBUG")
    >>> logger.debug("Verbose solver output")
    2025-11-02 10:30:47 - rkstiff.IF4 - DEBUG - Verbose solver output
    """
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)


def get_solver_logger(solver_class: type, loglevel: Union[str, int] = "WARNING") -> logging.Logger:
    r"""
    Return a standardized logger for a solver class.

    The logger name follows the pattern ``rkstiff.<ClassName>``,
    ensuring consistent log hierarchies across solvers (e.g. ``rkstiff.IF4``).

    Parameters
    ----------
    solver_class : type
        Class object (e.g. ``ETDRK4`` or ``IF4``). Its ``__name__`` attribute
        is used in constructing the logger name.
    loglevel : str or int, optional
        Logging level. Default is ``"WARNING"``.

    Returns
    -------
    logging.Logger
        Configured logger instance named ``"rkstiff.<ClassName>"``.

    Examples
    --------
    >>> class IF4:
    ...     pass
    >>> logger = get_solver_logger(IF4, "INFO")
    >>> logger.info("Solver initialized")
    2025-11-02 10:30:48 - rkstiff.IF4 - INFO - Solver initialized
    """
    logger_name = f"rkstiff.{solver_class.__name__}"
    return setup_logger(logger_name, loglevel)
