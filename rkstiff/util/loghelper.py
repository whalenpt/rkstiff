"""
Logging helper utilities for rkstiff solvers.

Provides functions for parsing log levels, setting up loggers, and configuring
logging for solver classes.
"""

import logging
from typing import Union


def _parse_loglevel(loglevel: Union[str, int]) -> int:
    """
    Convert loglevel input to logging integer constant.

    Parameters
    ----------
    loglevel : str or int
        Logging level as string (e.g., ``'INFO'``, ``'DEBUG'``) or integer
        constant (e.g., ``logging.INFO``).

    Returns
    -------
    int
        Numeric logging level corresponding to the input.

    Raises
    ------
    ValueError
        If `loglevel` is a string that does not match a valid logging level name.

    Examples
    --------
    >>> _parse_loglevel('INFO')
    20
    >>> _parse_loglevel(logging.DEBUG)
    10
    """
    if isinstance(loglevel, str):
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % loglevel)
        return numeric_level
    return loglevel


def get_level_name(level: int) -> str:
    """
    Get the string name for a logging level.

    Parameters
    ----------
    level : int
        Numeric logging level (e.g., ``10`` for DEBUG, ``20`` for INFO).

    Returns
    -------
    str
        Level name such as ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
        or ``'CRITICAL'``.

    Examples
    --------
    >>> get_level_name(logging.INFO)
    'INFO'
    """
    level_names = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    return level_names.get(level, "Level %s" % level)


def setup_logger(name: str, loglevel: Union[str, int] = "WARNING") -> logging.Logger:
    """
    Set up a logger with the specified name and level.

    Creates a new logger or retrieves an existing one, configures its logging
    level, and adds a StreamHandler with a formatted output if no handlers
    are already present. This prevents duplicate log messages.

    Parameters
    ----------
    name : str
        Name for the logger, typically ``__name__`` for module-level loggers
        or a class name for class-specific loggers.
    loglevel : str or int, optional
        Logging level. Can be specified as:

        - String: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``
        - Integer: ``logging.DEBUG``, ``logging.INFO``, etc.

        Default is ``'WARNING'``.

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use.

    Notes
    -----
    The logger uses the following format for messages::

        YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - message

    If the logger already has handlers, no additional handlers are added to
    avoid duplicate output.

    Examples
    --------
    >>> logger = setup_logger('my_module', 'INFO')
    >>> logger.info('This is an info message')
    2025-11-02 10:30:45 - my_module - INFO - This is an info message

    >>> logger = setup_logger('my_class', logging.DEBUG)
    >>> logger.debug('Debug information')
    2025-11-02 10:30:46 - my_class - DEBUG - Debug information
    """
    logger = logging.getLogger(name)

    # Set the logging level using helper
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)

    # Add handler if logger doesn't have one (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_log_level(logger: logging.Logger, loglevel: Union[str, int]) -> None:
    """
    Set the logging level for an existing logger.

    Updates the minimum severity level that the logger will process. Messages
    with severity below this level will be ignored.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to configure.
    loglevel : str or int
        Logging level as string (e.g., ``'INFO'``, ``'DEBUG'``) or integer
        constant (e.g., ``logging.INFO``).

    Raises
    ------
    ValueError
        If `loglevel` is a string that does not match a valid logging level name.

    See Also
    --------
    setup_logger : Create and configure a new logger.
    _parse_loglevel : Convert loglevel input to integer constant.

    Examples
    --------
    >>> logger = setup_logger('my_logger', 'WARNING')
    >>> set_log_level(logger, 'DEBUG')
    >>> logger.debug('Now visible')
    2025-11-02 10:30:47 - my_logger - DEBUG - Now visible
    """
    numeric_level = _parse_loglevel(loglevel)
    logger.setLevel(numeric_level)


def get_solver_logger(solver_class: type, loglevel: Union[str, int] = "WARNING") -> logging.Logger:
    """
    Create a logger for a solver instance.

    Creates a logger with a standardized name based on the solver class name.
    The logger name follows the pattern ``'rkstiff.<ClassName>'`` to maintain
    a consistent logging hierarchy.

    Parameters
    ----------
    solver_class : type
        The solver class (e.g., ``ETD35``, ``IF34``, ``ETDRK4``). The class's
        ``__name__`` attribute is used to construct the logger name.
    loglevel : str or int, optional
        Initial logging level. Can be specified as:

        - String: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``
        - Integer: ``logging.DEBUG``, ``logging.INFO``, etc.

        Default is ``'WARNING'``.

    Returns
    -------
    logging.Logger
        Configured logger for the solver with name ``'rkstiff.<ClassName>'``.

    See Also
    --------
    setup_logger : Lower-level function for creating loggers.

    Examples
    --------
    >>> class ETD35:
    ...     pass
    >>> logger = get_solver_logger(ETD35, 'INFO')
    >>> logger.name
    'rkstiff.ETD35'
    >>> logger.info('Solver initialized')
    2025-11-02 10:30:48 - rkstiff.ETD35 - INFO - Solver initialized
    """
    logger_name = "rkstiff.%s" % solver_class.__name__
    return setup_logger(logger_name, loglevel)
