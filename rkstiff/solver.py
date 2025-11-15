r"""
Base solver infrastructure for all rkstiff PDE solvers
==================================================================

Base solver infrastructure for exponential time-differencing (ETD)
and related stiff integrators.

This module defines the :class:`BaseSolver` abstract base class, which
provides common initialization, logging, and validation logic for all
solver subclasses (e.g., :class:`rkstiff.etd4.ETD4`,
:class:`rkstiff.etd5.ETD5`).

It standardizes handling of the linear operator :math:`\mathcal{L}`,
the nonlinear function :math:`\mathcal{N}(\mathbf{U})`, and logging
verbosity across all stiff PDE solvers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Union, Literal
import numpy as np
from typing import TYPE_CHECKING

from .util.loghelper import get_solver_logger, set_log_level, get_level_name

if TYPE_CHECKING:
    from .util.solver_type import SolverType


class BaseSolver(ABC):
    r"""
    Abstract base class for all stiff solvers.

    Provides shared functionality for initializing linear and nonlinear
    components, configuring logging, and managing solution state.

    Subclasses (such as :class:`ETD4` or :class:`ETD5`) implement
    specific numerical time-stepping schemes by defining the abstract
    methods :meth:`reset` and :meth:`_reset`.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}` defining the stiff part of
        the system. Can be either:

        * 1D array — representing a diagonal operator
        * 2D square array — representing a full linear operator matrix
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})` returning
        the nonlinear contribution for a given solution vector.
    loglevel : str or int, optional
        Logging level. Accepts either a string
        (``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
        ``'CRITICAL'``) or the corresponding integer constant from
        :mod:`logging`. Default is ``'WARNING'``.

    Attributes
    ----------
    lin_op : np.ndarray
        Linear operator associated with the stiff system.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    logger : logging.Logger
        Logger instance configured for the solver.
    t : list of float
        Time points generated during the last call to :meth:`evolve`.
    u : list of np.ndarray
        Solution vectors corresponding to time points in :attr:`t`.
    _diag : bool
        Indicates whether the linear operator is diagonal (True) or
        full matrix (False).

    Raises
    ------
    ValueError
        If ``lin_op`` is not 1D or a 2D square matrix.

    Notes
    -----
    This base class is not meant to be instantiated directly.
    Derived solvers should inherit from :class:`BaseSolver` and
    implement both :meth:`reset` and :meth:`_reset` to define
    problem-specific initialization and internal state clearing.

    .. tip::
       The base class automatically detects whether ``lin_op`` is
       diagonal, allowing subclasses to optimize accordingly.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize a base solver instance.

        Sets up internal attributes, validates the shape of the linear
        operator, and configures logging behavior.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator. Must be 1D (diagonal) or a 2D square matrix.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function mapping the solution vector to its
            nonlinear component.
        loglevel : str or int, optional
            Logging verbosity level. Default is ``'WARNING'``.

        Raises
        ------
        ValueError
            If ``lin_op`` is not 1D or not a square 2D matrix.
        """
        self.lin_op = lin_op
        self.nl_func = nl_func

        # Create and configure logger
        self.logger = get_solver_logger(self.__class__, loglevel)
        self.logger.info("Initialized %s solver", self.__class__.__name__)

        # Storage for time evolution
        self.t, self.u = [], []

        # Validate shape and determine if diagonal
        dims = lin_op.shape
        if len(dims) not in (1, 2):
            raise ValueError("lin_op must be 1D or 2D")
        if len(dims) == 2 and dims[0] != dims[1]:
            raise ValueError("lin_op must be a square matrix")

        self._diag = len(dims) == 1
        self.logger.debug("Linear operator shape: %s, diagonal: %s", dims, self._diag)

    @property
    @abstractmethod
    def solver_type(self) -> SolverType:
        """
        Return the type of this solver.

        Returns
        -------
        SolverType
            Either CONSTANT_STEP or ADAPTIVE_STEP.

        Notes
        -----
        This is an abstract property that must be overridden in subclasses
        BaseSolverCS and BaseSolverAS.

        Examples
        --------
        >>> from rkstiff.if4 import IF4
        >>> solver = IF4(lin_op, nl_func)
        >>> solver.solver_type
        <SolverType.CONSTANT_STEP: 1>
        """

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def set_loglevel(self, loglevel: Union[str, int]) -> None:
        """
        Adjust the solver's logging verbosity at runtime.

        Parameters
        ----------
        loglevel : str or int
            New logging level. Accepts standard string levels or numeric
            constants from :mod:`logging`.

        Examples
        --------
        >>> solver.set_loglevel("INFO")
        >>> solver.set_loglevel(logging.DEBUG)
        """
        set_log_level(self.logger, loglevel)
        self.logger.info("Log level changed to %s", get_level_name(self.logger.level))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the solver to its initial state.

        Clears stored time and solution arrays, restoring the solver
        to its post-initialization condition. Typically invoked before
        restarting a simulation or integrating a new problem.

        Notes
        -----
        Subclasses should call ``super().reset()`` if extending this
        method, and implement :meth:`_reset` to clear any
        solver-specific caches.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        Reset solver-specific internal state.

        Must be implemented by subclasses to clear any cached
        coefficients, temporary arrays, or integration-specific
        parameters.

        This method is typically invoked by :meth:`reset` and should
        not be called directly by users.
        """
