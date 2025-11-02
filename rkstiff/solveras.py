"""
rkstiff.solveras

Provides base class for adaptive-step solvers
"""

from abc import abstractmethod
from typing import Tuple, Optional, Callable, Union, Literal
import numpy as np
from .solver import BaseSolver


class SolverConfig:
    """
    Configuration parameters for adaptive-step stiff solvers.

    Parameters
    ----------
    epsilon : float, optional
        Relative error tolerance for adaptive stepping. Default is 1e-4.
    incr_f : float, optional
        Increment factor for adaptive step sizing (must be > 1.0). Default is 1.25.
    decr_f : float, optional
        Decrement factor for adaptive step sizing (must be < 1.0). Default is 0.85.
    safety_f : float, optional
        Safety factor for adaptive stepping (must be <= 1.0). Default is 0.8.
    adapt_cutoff : float, optional
        Cutoff threshold for adaptive step size computation (must be < 1.0). Default is 0.01.
        Modes with relative magnitude below this threshold are ignored.
    minh : float, optional
        Minimum allowable step size. Default is 1e-16.
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        incr_f: float = 1.25,
        decr_f: float = 0.85,
        safety_f: float = 0.8,
        adapt_cutoff: float = 0.01,
        minh: float = 1e-16,
    ) -> None:
        """
        Initialize SolverConfig with validated parameters.

        Parameters
        ----------
        epsilon : float, optional
            Relative error tolerance for adaptive stepping.
        incr_f : float, optional
            Increment factor for adaptive step sizing.
        decr_f : float, optional
            Decrement factor for adaptive step sizing.
        safety_f : float, optional
            Safety factor for adaptive stepping.
        adapt_cutoff : float, optional
            Cutoff threshold for adaptive step size computation.
        minh : float, optional
            Minimum allowable step size.
        """
        self._epsilon = None
        self._incr_f = None
        self._decr_f = None
        self._safety_f = None
        self._adapt_cutoff = None
        self._minh = None

        # Use setters for validation
        self.epsilon = epsilon
        self.incr_f = incr_f
        self.decr_f = decr_f
        self.safety_f = safety_f
        self.adapt_cutoff = adapt_cutoff
        self.minh = minh

    # ---- epsilon ----
    @property
    def epsilon(self) -> float:
        """Relative error tolerance for adaptive stepping."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set relative error tolerance, must be positive."""
        if value <= 0:
            raise ValueError("epsilon must be positive but is %s" % value)
        self._epsilon = float(value)

    # ---- incr_f ----
    @property
    def incr_f(self) -> float:
        """Increment factor for adaptive step sizing."""
        return self._incr_f

    @incr_f.setter
    def incr_f(self, value: float) -> None:
        """Set increment factor, must be greater than 1.0."""
        if value <= 1.0:
            raise ValueError("incr_f must be > 1.0 but is %s" % value)
        self._incr_f = float(value)

    # ---- decr_f ----
    @property
    def decr_f(self) -> float:
        """Decrement factor for adaptive step sizing."""
        return self._decr_f

    @decr_f.setter
    def decr_f(self, value: float) -> None:
        """Set decrement factor, must be less than 1.0."""
        if value >= 1.0:
            raise ValueError("decr_f must be < 1.0 but is %s" % value)
        self._decr_f = float(value)

    # ---- safety_f ----
    @property
    def safety_f(self) -> float:
        """Safety factor for adaptive stepping."""
        return self._safety_f

    @safety_f.setter
    def safety_f(self, value: float) -> None:
        """Set safety factor, must be less than or equal to 1.0."""
        if value > 1.0:
            raise ValueError("safety_f must be <= 1.0 but is %s" % value)
        self._safety_f = float(value)

    # ---- adapt_cutoff ----
    @property
    def adapt_cutoff(self) -> float:
        """Cutoff threshold for adaptive step size computation."""
        return self._adapt_cutoff

    @adapt_cutoff.setter
    def adapt_cutoff(self, value: float) -> None:
        """Set adapt_cutoff, must be less than 1.0."""
        if value >= 1.0:
            raise ValueError("adapt_cutoff must be < 1.0 but is %s" % value)
        self._adapt_cutoff = float(value)

    # ---- minh ----
    @property
    def minh(self) -> float:
        """Minimum allowable step size."""
        return self._minh

    @minh.setter
    def minh(self, value: float) -> None:
        """Set minimum step size, must be positive."""
        if value <= 0:
            raise ValueError("minh must be positive but is %s" % value)
        self._minh = float(value)


class StiffSolverAS(BaseSolver):
    """
    Base class for adaptive-step Runge-Kutta solvers for stiff systems.

    Solves systems of the form dU/dt = L*U + NL(U), where L is a linear
    operator and NL is a nonlinear function. The solver automatically adjusts
    the time step to maintain a specified error tolerance.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator L. Can be 1D (diagonal) or 2D square matrix.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function nl_func(U).
    config : SolverConfig, optional
        Configuration parameters for adaptive stepping.
    loglevel : str or int, optional
        Logging level. Can be:
        - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        - Integer: logging.DEBUG, logging.INFO, etc.
        - Default: 'WARNING'

    Attributes
    ----------
    config : SolverConfig
        Configuration parameters for adaptive stepping.
    epsilon : float
        Relative error tolerance.
    incr_f : float
        Increment factor for step size increases (> 1.0).
    decr_f : float
        Decrement factor for step size decreases (< 1.0).
    safety_f : float
        Safety factor for adaptive stepping (<= 1.0).
    adapt_cutoff : float
        Threshold for ignoring small modes (< 1.0).
    minh : float
        Minimum allowable step size.

    Raises
    ------
    ValueError
        If lin_op is not 1D or 2D square, or if config parameters are invalid.

    Notes
    -----
    Subclasses must implement _reset(), _update_stages(), and _q() methods.
    """

    class SolverError(RuntimeError):
        """Base exception for solver failures."""

    class MaxLoopsExceeded(SolverError):
        """Raised when adaptive step exceeds maximum allowed attempts."""

    class MinimumStepReached(SolverError):
        """Raised when step size reaches minimum allowed value."""

    MAX_LOOPS = 50
    MAX_S = 4.0  # Maximum step increase factor
    MIN_S = 0.25  # Minimum step decrease factor

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize the adaptive-step solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator. Must be 1D (diagonal) or 2D square matrix.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function mapping state vector to nonlinear contribution.
        config : SolverConfig, optional
            Configuration with epsilon, incr_f, decr_f, safety_f, adapt_cutoff, minh.
        loglevel : str or int, optional
            Logging level. Default is 'WARNING'.

        Raises
        ------
        ValueError
            If lin_op has invalid dimensions or if config parameters violate constraints.
        """
        super().__init__(lin_op, nl_func, loglevel)

        self.config = config
        self.logger.debug(
            "Config: epsilon=%s, incr_f=%s, decr_f=%s, safety_f=%s",
            config.epsilon,
            config.incr_f,
            config.decr_f,
            config.safety_f,
        )

        self.__t0, self.__tf, self.__tc = 0, 0, 0
        self._accept = False

    def reset(self) -> None:
        """
        Reset solver to initial state.

        Clears stored time points and solution arrays. Prepares
        solver for a new call to evolve() or step() with fresh initial conditions.
        """
        self.logger.debug("Resetting solver state")
        self.t, self.u = [], []
        self.__t0, self.__tf, self.__tc = 0, 0, 0
        self._accept = False
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        """Reset solver-specific internal state. Must be implemented by subclasses."""

    @abstractmethod
    def _update_stages(self, u: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray]
            Next state vector and local error estimate.
        """

    @abstractmethod
    def _q(self):
        """Return the order parameter q for step size computation."""

    def step(self, u: np.ndarray, h_suggest: float) -> Tuple[np.ndarray, float, float]:
        """
        Propagate solution by one adaptive time step.

        Attempts to advance the solution using the suggested step size,
        automatically reducing it if the error exceeds tolerance.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h_suggest : float
            Suggested time step size (may be reduced to meet tolerance).

        Returns
        -------
        unew : np.ndarray
            Updated state vector after one accepted step.
        h : float
            Actual step size taken (may be less than h_suggest).
        h_suggest : float
            Suggested step size for next step.

        Raises
        ------
        MaxLoopsExceeded
            If too many attempts are needed to find an acceptable step size.
        MinimumStepReached
            If step size falls below minimum allowed value.
        """
        h = h_suggest
        assert h >= 0.0
        self.logger.debug("Starting step with h_suggest=%s", h_suggest)

        numloops = 0
        while True:
            unew, err = self._update_stages(u, h)
            # Compute step size change factor s
            s = self._compute_s(unew, err)
            self.logger.debug("Computed s=%s for h=%s", s, h)

            # If s is less than 1, inf, or nan, reject step and reduce step size
            if np.isinf(s) or np.isnan(s) or s < 1.0:
                h = self._reject_step_size(s, h)
            # If s is bigger than 1 accept h and the step
            else:
                h_suggest = self._accept_step_size(s, h)
                self.logger.debug("Step accepted, returning h=%s, h_suggest=%s", h, h_suggest)
                return unew, h, h_suggest

            numloops += 1
            if numloops > self.MAX_LOOPS:
                failure_str = (
                    "Solver failed: adaptive step made too many attempts to find a step "
                    "size with an acceptible amount of error."
                )
                self.logger.error(failure_str)
                raise self.MaxLoopsExceeded(failure_str)
            if h < self.config.minh:
                failure_str = "Solver failed: adaptive step reached minimum step size"
                self.logger.error(failure_str)
                raise self.MinimumStepReached(failure_str)

    def _compute_s(self, u: np.ndarray, err: np.ndarray) -> float:
        """
        Compute step size adjustment factor.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        err : np.ndarray
            Local error estimate.

        Returns
        -------
        float
            Step size adjustment factor s. Values > 1 indicate step acceptance.
        """
        # Use adapt_cutoff to ignore small modes/values in the computation of the step size
        magu = np.abs(u)
        idx = magu / magu.max() > self.config.adapt_cutoff
        tol = self.config.epsilon * np.linalg.norm(u[idx])
        s = self.config.safety_f * np.power(tol / np.linalg.norm(err[idx]), 1.0 / self._q())
        return s

    def _reject_step_size(self, s: float, h: float) -> float:
        """
        Compute reduced step size after rejection.

        Parameters
        ----------
        s : float
            Step size adjustment factor.
        h : float
            Current step size.

        Returns
        -------
        float
            New (reduced) step size.
        """
        self._accept = False
        # Check that s is a number
        if np.isinf(s) or np.isnan(s):
            msg = "inf or nan number encountered: reducing step size to %s" % h
            self.logger.warning(msg)
            return self.MIN_S * h

        s = np.max([s, self.MIN_S])  # dont let s be too small
        s = np.min([s, self.config.decr_f])  # dont let s be too close to 1
        msg = "step rejected with s = %.2f" % s
        self.logger.debug(msg)
        hnew = s * h
        msg = "reducing step size to %s" % hnew
        self.logger.debug(msg)
        return hnew

    def _accept_step_size(self, s: float, h: float) -> float:
        """
        Compute suggested step size after acceptance.

        Parameters
        ----------
        s : float
            Step size adjustment factor.
        h : float
            Current step size.

        Returns
        -------
        float
            Suggested step size for next step.
        """
        self._accept = True
        s = np.min([s, self.MAX_S])  # dont let s be too big
        msg = "step accepted with s = %.2f" % s
        self.logger.debug(msg)

        # if s much larger than 1, increase the step size
        if s > self.config.incr_f:
            h_suggest = s * h
            msg = "increasing step size to %s" % h_suggest
            self.logger.debug(msg)
            return h_suggest
        return h

    def evolve(
        self,
        u: np.ndarray,
        t0: float,
        tf: float,
        h_init: Optional[float] = None,
        store_data: bool = True,
        store_freq: int = 1,
    ) -> np.ndarray:
        """
        Evolve solution from initial to final time using adaptive stepping.

        Parameters
        ----------
        u : np.ndarray
            Initial state vector at time t0.
        t0 : float
            Initial time.
        tf : float
            Final time.
        h_init : float, optional
            Initial step size. Defaults to (tf - t0) / 100 if not specified.
        store_data : bool, default=True
            Whether to store intermediate time points and solutions in
            self.t and self.u.
        store_freq : int, default=1
            Store data every store_freq accepted steps.

        Returns
        -------
        np.ndarray
            Final state vector at time tf.

        Notes
        -----
        Stored data is accessible via self.t (times) and self.u (states).
        """
        self.reset()
        self.logger.info("Starting evolution from t=%s to t=%s", t0, tf)
        self.__t0, self.__tf, self.__tc = t0, tf, t0

        if store_data:
            self.t.append(t0)
            self.u.append(u)

        # Set initial step size if none given
        if h_init is None:
            h_init = (self.__tf - self.__t0) / 100.0
        h = h_init
        self.logger.debug("Initial step size h=%s, store_freq=%s", h, store_freq)

        # Make sure step size isn't larger than entire propagation time
        if self.__tc + h > self.__tf:
            h = self.__tf - self.__tc

        step_count = 0
        while self.__tc < self.__tf:
            u, h, h_suggest = self.step(u, h)
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

            if self.__tc + h_suggest > self.__tf:
                h = self.__tf - self.__tc
            else:
                h = h_suggest

            if store_data and (step_count % store_freq == 0):
                self.t.append(self.__tc)
                self.u.append(u)
                self.logger.debug("Stored solution at t=%.6f (step %d)", self.__tc, step_count)

        self.logger.info("Evolution complete after %d steps", step_count)
        self.logger.info("Stored %d solution snapshots", len(self.u))
        return u
