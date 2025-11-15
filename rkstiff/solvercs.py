r"""
Base solver infrastructure for rkstiff constant step PDE solvers
================================================================

Constant-step solver infrastructure for stiff time-dependent systems.

This module defines :class:`BaseSolverCS`, a foundation for solvers that
advance semi-linear differential equations using *fixed* time steps:

.. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is a (possibly stiff) linear operator and
:math:`\mathcal{N}` is a nonlinear function. Subclasses such as ETD4 or ETD5
implement constant-step exponential Runge-Kutta methods based on this interface.
"""

from abc import abstractmethod
from typing import Callable, Union, Literal
import numpy as np
from .solver import BaseSolver
from .util.solver_type import SolverType


# ======================================================================
# Base Class
# ======================================================================
class BaseSolverCS(BaseSolver):
    r"""
    Abstract base class for constant-step stiff solvers.

    Provides the common structure for fixed-step exponential
    Runge-Kutta and related integrators. Derived classes implement
    stage updates and solver-specific initialization.

    The governing semi-linear equation is

    .. math::

            \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` represents the linear (stiff) operator and
    :math:`\mathcal{N}` the nonlinear component.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}`. Must be 1D (diagonal)
        or a 2D square matrix.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    loglevel : str or int, default='WARNING'
        Logging verbosity. Accepts string names (``'DEBUG'``, ``'INFO'``…)
        or numeric :mod:`logging` constants.

    Attributes
    ----------
    lin_op : np.ndarray
        Linear operator passed at construction.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function for the system.
    t : list[float]
        Time points recorded during evolution.
    u : list[np.ndarray]
        Solution vectors recorded during evolution.

    Raises
    ------
    ValueError
        If ``lin_op`` is not 1D or a square 2D matrix.

    Notes
    -----
    - Subclasses must implement :meth:`_reset` and :meth:`_update_stages`.
    - The step size remains constant throughout evolution.
    - For adaptive time-stepping, use :class:`rkstiff.solveras.BaseSolverAS`.

    References
    ----------
    P. Whalen, M. Brio, and J. V. Moloney,
    *Exponential time-differencing with embedded Runge-Kutta adaptive step control*,
    J. Comput. Phys. 280 (2015), 579–601.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """Initialize a constant-step solver and validate inputs."""
        super().__init__(lin_op, nl_func, loglevel)
        self.__tf, self.__tc = 0, 0

    @property
    def solver_type(self) -> SolverType:
        """
        Return the solver type for constant-step solvers.
        
        Returns
        -------
        SolverType
            Always returns ``SolverType.CONSTANT_STEP``.
            
        Examples
        --------
        >>> from rkstiff.if4 import IF4
        >>> solver = IF4(lin_op, nl_func)
        >>> solver.solver_type == SolverType.CONSTANT_STEP
        True
        """
        return SolverType.CONSTANT_STEP

    # ------------------------------------------------------------------
    # Reset logic
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset solver state and clear stored time/solution data."""
        self.logger.debug("Resetting constant-step solver state")
        self.t, self.u = [], []
        self.__tf, self.__tc = 0, 0
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        """Clear subclass-specific caches or precomputed data."""

    # ------------------------------------------------------------------
    # Stage update interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Advance one step of size :math:`h`.

        Subclasses must implement this method to perform the internal
        Runge-Kutta or ETD stage updates.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector :math:`\mathbf{u}_n`.
        h : float
            Constant time step size :math:`\Delta t`.

        Returns
        -------
        np.ndarray
            Updated state :math:`\mathbf{u}_{n+1}`.
        """

    # ------------------------------------------------------------------
    # Step routine
    # ------------------------------------------------------------------
    def step(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Perform a single constant-step propagation.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Constant step size (must be non-negative).

        Returns
        -------
        np.ndarray
            Updated solution after one full time step.

        Notes
        -----
        This method simply wraps :meth:`_update_stages` and performs
        minimal validation and logging.
        """
        assert h >= 0.0
        self.logger.debug("Executing constant step with h=%s", h)
        return self._update_stages(u, h)

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    def evolve(
        self,
        u: np.ndarray,
        t0: float,
        tf: float,
        h: float,
        store_data: bool = True,
        store_freq: int = 1,
    ) -> np.ndarray:
        r"""
        Integrate the system from :math:`t_0` to :math:`t_f` using fixed step size.

        Repeatedly applies :meth:`step` with constant :math:`h`
        until the final time is reached.

        Parameters
        ----------
        u : np.ndarray
            Initial solution vector at :math:`t_0`.
        t0 : float
            Initial time.
        tf : float
            Final time (integration stops when ``t ≥ tf``).
        h : float
            Constant step size :math:`\Delta t`.
        store_data : bool, default=True
            Whether to store intermediate results in :attr:`t` and :attr:`u`.
        store_freq : int, default=1
            Frequency of storing data; every ``store_freq`` steps.

        Returns
        -------
        np.ndarray
            Final solution vector at :math:`t_f`.

        Raises
        ------
        ValueError
            If ``h`` exceeds the total time span ``tf - t0``.

        Notes
        -----
        - The time grid is uniformly spaced with spacing :math:`h`.
        - Stored data can be accessed through :attr:`t` and :attr:`u`.

        Example
        -------
        >>> solver = ETD4(lin_op, nl_func)
        >>> u_final = solver.evolve(u0, t0=0.0, tf=10.0, h=0.05)
        >>> len(solver.t)
        200
        """
        self.reset()
        self.logger.info("Starting constant-step evolution from t=%s to t=%s", t0, tf)
        self.__tf, self.__tc = tf, t0

        if store_data:
            self.t.append(t0)
            self.u.append(u)

        # Ensure step size is valid
        if self.__tc + h > self.__tf:
            raise ValueError("Step size h must be <= (tf - t0); reduce h or extend tf.")

        self.logger.debug("Step size h=%s, store_freq=%s", h, store_freq)

        step_count = 0
        while self.__tc < self.__tf:
            u = self.step(u, h)
            self.__tc += h
            step_count += 1

            if step_count % 100 == 0:
                self.logger.info(
                    "Progress: t=%.6f/%.6f (%.1f%%), steps=%d",
                    self.__tc,
                    self.__tf,
                    100 * self.__tc / self.__tf,
                    step_count,
                )

            if store_data and (step_count % store_freq == 0):
                self.t.append(self.__tc)
                self.u.append(u)
                self.logger.debug("Stored snapshot at t=%.6f (step %d)", self.__tc, step_count)

        self.logger.info("Evolution complete after %d steps", step_count)
        self.logger.info("Stored %d snapshots", len(self.u))
        return u
