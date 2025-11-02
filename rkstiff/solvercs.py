"""
rkstiff.solvercs

Provides base classes for constant-step stiff solvers.
"""

from abc import abstractmethod
from typing import Callable, Union, Literal
import numpy as np
from .solver import BaseSolver


class StiffSolverCS(BaseSolver):
    """
    Base class for constant-step Runge-Kutta solvers for stiff systems.

    Solves systems of the form dU/dt = L*U + NL(U), where L is a linear
    operator and nl_func is a nonlinear function, using a fixed time step.

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
    All attributes from BaseSolver

    Raises
    ------
    ValueError
        If lin_op is not 1D or 2D square.

    Notes
    -----
    Subclasses must implement _reset() and _update_stages() methods.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize the constant-step solver.

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
        super().__init__(lin_op, nl_func, loglevel)
        self.__tf, self.__tc = 0, 0

    def reset(self) -> None:
        """
        Reset solver to initial state.

        Clears stored time points and solution arrays. Prepares
        solver for a new call to evolve() or step() with fresh initial conditions.
        """
        self.logger.debug("Resetting solver state")
        self.t, self.u = [], []
        self.__tf, self.__tc = 0, 0
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        """Reset solver-specific internal state. Must be implemented by subclasses."""

    @abstractmethod
    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance solution by one time step.

        Must be implemented by subclasses to perform the RK stage updates.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time step size.

        Returns
        -------
        np.ndarray
            Next state vector.
        """

    def step(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Propagate solution by one constant time step.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time step size (must be >= 0).

        Returns
        -------
        np.ndarray
            Updated state vector after one step.
        """
        assert h >= 0.0
        self.logger.debug("Step at h=%s", h)
        unew = self._update_stages(u, h)
        return unew

    def evolve(
        self,
        u: np.ndarray,
        t0: float,
        tf: float,
        h: float,
        store_data: bool = True,
        store_freq: int = 1,
    ) -> np.ndarray:
        """
        Evolve solution from initial to final time using constant stepping.

        Parameters
        ----------
        u : np.ndarray
            Initial state vector at time t0.
        t0 : float
            Initial time.
        tf : float
            Final time (actual final time may exceed this slightly).
        h : float
            Constant time step size.
        store_data : bool, default=True
            Whether to store intermediate time points and solutions in
            self.t and self.u.
        store_freq : int, default=1
            Store data every store_freq steps.

        Returns
        -------
        np.ndarray
            Final state vector at or after time tf.

        Raises
        ------
        ValueError
            If step size h is greater than (tf - t0).

        Notes
        -----
        Stored data is accessible via self.t (times) and self.u (states).
        """
        self.reset()
        self.logger.info("Starting evolution from t=%s to t=%s", t0, tf)
        self.__tf, self.__tc = tf, t0

        if store_data:
            self.t.append(t0)
            self.u.append(u)

        # Make sure step size isn't larger than entire propagation time
        if self.__tc + h > self.__tf:
            raise ValueError("Reduce step size h, it needs to be less than or equal to tf - t0")

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
                self.logger.debug("Stored solution at t=%.6f (step %d)", self.__tc, step_count)

        self.logger.info("Evolution complete after %d steps", step_count)
        self.logger.info("Stored %d solution snapshots", len(self.u))
        return u
