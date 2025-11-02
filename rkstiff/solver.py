"""
rkstiff.solver

Provides base class for all solvers
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, Literal
import numpy as np
from .util.loghelper import get_solver_logger, set_log_level, get_level_name

class BaseSolver(ABC):
    """
    Base class for all stiff solvers.

    Provides common functionality for initializing linear operators,
    nonlinear functions, and logging configuration.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator L. Can be 1D (diagonal) or 2D square matrix.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function NL(U).
    loglevel : str or int, optional
        Logging level. Can be:
        - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        - Integer: logging.DEBUG, logging.INFO, etc.
        - Default: 'WARNING'

    Attributes
    ----------
    lin_op : np.ndarray
        Linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
    logger : logging.Logger
        Logger instance for this solver.
    t : list
        Time points from most recent evolve() call.
    u : list
        Solution arrays from most recent evolve() call.

    Raises
    ------
    ValueError
        If lin_op is not 1D or 2D square.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize the base solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator. Must be 1D (diagonal) or 2D square matrix.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function mapping state vector to nonlinear contribution.
        loglevel : str or int, optional
            Logging level. Default is 'WARNING'.

        Raises
        ------
        ValueError
            If lin_op has invalid dimensions.
        """
        self.lin_op = lin_op
        self.nl_func = nl_func

        # Create logger using helper function
        self.logger = get_solver_logger(self.__class__, loglevel)
        self.logger.info("Initialized %s solver", self.__class__.__name__)

        # Initialize storage
        self.t, self.u = [], []

        # Validate and set operator properties
        dims = lin_op.shape
        self._diag = True
        if len(dims) > 2 or len(dims) == 0:
            raise ValueError("lin_op must be a 1D or 2D array")
        if len(dims) == 2:
            if dims[0] != dims[1]:
                raise ValueError("lin_op must be a square matrix")
            self._diag = False

        self.logger.debug("Linear operator shape: %s, diagonal: %s", dims, self._diag)

    def set_loglevel(self, loglevel: Union[str, int]) -> None:
        """
        Change the logging level after initialization.

        Parameters
        ----------
        loglevel : str or int
            New logging level
        """
        set_log_level(self.logger, loglevel)
        self.logger.info("Log level changed to %s", get_level_name(self.logger.level))

    @abstractmethod
    def reset(self) -> None:
        """
        Reset solver to initial state.

        Clears stored time points and solution arrays. Prepares
        solver for a new call to evolve() or step() with fresh initial conditions.
        """

    @abstractmethod
    def _reset(self) -> None:
        """Reset solver-specific internal state. Must be implemented by subclasses."""
