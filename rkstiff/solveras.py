r"""
Base solver infrastructure for rkstiff adaptive step PDE solvers
================================================================

Adaptive-step solver infrastructure for stiff time-dependent systems.

This module defines the :class:`BaseSolverAS` abstract class and its
associated :class:`SolverConfig`, which provide the foundation for
adaptive time-stepping solvers such as ETD(3,5) and other embedded
Runge–Kutta–type integrators.

Adaptive solvers automatically adjust their time step to maintain
a user-specified accuracy tolerance, based on local error estimates
computed during integration.
"""

from abc import abstractmethod
from typing import Tuple, Optional, Callable, Union, Literal
import numpy as np
from .solver import BaseSolver
from .util.solver_type import SolverType


# ======================================================================
# Solver Configuration
# ======================================================================
class SolverConfig:
    r"""
    Configuration parameters for adaptive-step stiff solvers.

    Defines numerical tolerances and scaling factors that control
    adaptive time step selection. Used by all adaptive solvers derived
    from :class:`BaseSolverAS`.

    Parameters
    ----------
    epsilon : float, default=1e-4
        Relative error tolerance for adaptive stepping. Smaller values
        yield higher accuracy but may reduce efficiency.
    incr_f : float, default=1.25
        Step-size increase factor applied after successful steps.
        Must satisfy ``incr_f > 1.0``.
    decr_f : float, default=0.85
        Step-size reduction factor applied after rejected steps.
        Must satisfy ``decr_f < 1.0``.
    safety_f : float, default=0.8
        Safety factor for damping overly aggressive changes in step size.
        Must satisfy ``safety_f <= 1.0``.
    adapt_cutoff : float, default=0.01
        Relative magnitude threshold for excluding small modes when
        computing adaptive step corrections.
    minh : float, default=1e-16
        Minimum allowable step size.

    Raises
    ------
    ValueError
        If any parameter violates its constraint.

    Notes
    -----
    The combination of ``epsilon``, ``safety_f``, and ``adapt_cutoff``
    primarily determines the balance between stability and adaptivity.
    Typical recommended settings are:

    .. code-block:: python

        SolverConfig(epsilon=1e-5, safety_f=0.9, adapt_cutoff=0.001)
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
        """Initialize and validate adaptive solver configuration."""
        self._epsilon = None
        self._incr_f = None
        self._decr_f = None
        self._safety_f = None
        self._adapt_cutoff = None
        self._minh = None

        # Use property setters for validation
        self.epsilon = epsilon
        self.incr_f = incr_f
        self.decr_f = decr_f
        self.safety_f = safety_f
        self.adapt_cutoff = adapt_cutoff
        self.minh = minh

    # ------------------------------------------------------------------
    # Properties (documented for autodoc clarity)
    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Relative error tolerance for adaptive stepping."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set relative error tolerance, must be positive."""
        if value <= 0:
            raise ValueError(f"epsilon must be positive but is {value}")
        self._epsilon = float(value)

    @property
    def incr_f(self) -> float:
        """Increment factor for adaptive step sizing (must be > 1.0)."""
        return self._incr_f

    @incr_f.setter
    def incr_f(self, value: float) -> None:
        """Set increment factor, must be greater than 1.0."""
        if value <= 1.0:
            raise ValueError(f"incr_f must be > 1.0 but is {value}")
        self._incr_f = float(value)

    @property
    def decr_f(self) -> float:
        """Decrement factor for adaptive step sizing (must be < 1.0)."""
        return self._decr_f

    @decr_f.setter
    def decr_f(self, value: float) -> None:
        """Set decrement factor, must be less than 1.0."""
        if value >= 1.0:
            raise ValueError(f"decr_f must be < 1.0 but is {value}")
        self._decr_f = float(value)

    @property
    def safety_f(self) -> float:
        """Safety factor for adaptive stepping (must be ≤ 1.0)."""
        return self._safety_f

    @safety_f.setter
    def safety_f(self, value: float) -> None:
        """Set safety factor, must be ≤ 1.0."""
        if value > 1.0:
            raise ValueError(f"safety_f must be <= 1.0 but is {value}")
        self._safety_f = float(value)

    @property
    def adapt_cutoff(self) -> float:
        """Cutoff threshold for ignoring small modes during adaptivity."""
        return self._adapt_cutoff

    @adapt_cutoff.setter
    def adapt_cutoff(self, value: float) -> None:
        """Set adaptivity cutoff, must be less than 1.0."""
        if value >= 1.0:
            raise ValueError(f"adapt_cutoff must be < 1.0 but is {value}")
        self._adapt_cutoff = float(value)

    @property
    def minh(self) -> float:
        """Minimum allowable step size."""
        return self._minh

    @minh.setter
    def minh(self, value: float) -> None:
        """Set minimum step size, must be positive."""
        if value <= 0:
            raise ValueError(f"minh must be positive but is {value}")
        self._minh = float(value)


# ======================================================================
# Adaptive Solver Base Class
# ======================================================================
class BaseSolverAS(BaseSolver):
    r"""
    Abstract base class for adaptive-step stiff solvers.

    Provides an adaptive time-stepping framework for integrating
    semi-linear systems of the form

    .. math::

            \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` is a (possibly stiff) linear operator
    and :math:`\mathcal{N}` is a nonlinear function.

    Subclasses must implement :meth:`_update_stages` (to perform the
    embedded Runge–Kutta stage updates), :meth:`_reset` (to clear solver
    state), and :meth:`_q` (to define the order of accuracy for step
    size control).

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}` defining the stiff part of the system.
        Can be either 1D (diagonal) or a full 2D matrix.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    config : SolverConfig, optional
        Configuration object specifying tolerances and scaling factors
        for adaptive step control.
    loglevel : str or int, default='WARNING'
        Logging level for runtime diagnostics.

    Attributes
    ----------
    config : SolverConfig
        Adaptive step configuration (tolerances and safety parameters).
    epsilon : float
        Relative error tolerance (:attr:`config.epsilon`).
    minh : float
        Minimum allowable step size (:attr:`config.minh`).
    _accept : bool
        Flag indicating whether the most recent step was accepted.

    Raises
    ------
    ValueError
        If ``lin_op`` has invalid dimensions or configuration parameters are inconsistent.

    Notes
    -----
    Adaptive time-stepping uses local error control via embedded Runge–Kutta
    pairs, dynamically adjusting :math:`h` such that

    .. math::

        \| \mathbf{e}_n \| \leq \varepsilon \, \| \mathbf{u}_n \|,

    where :math:`\mathbf{e}_n` is the local truncation error and
    :math:`\varepsilon` is the tolerance.

    .. tip::
       Implementations typically define `_update_stages()` to return both
       a candidate solution and an error estimate, which are used to adjust
       the next step size.
    """

    # ------------------------------------------------------------------
    # Internal exceptions
    # ------------------------------------------------------------------
    class SolverError(RuntimeError):
        """Base exception for solver-related runtime errors."""

    class MaxLoopsExceeded(SolverError):
        """Raised when too many attempts are made to find a valid adaptive step."""

    class MinimumStepReached(SolverError):
        """Raised when the adaptive step size falls below the minimum allowed value."""

    MAX_LOOPS = 50  #: Maximum retry attempts per adaptive step
    MAX_S = 4.0  #: Maximum allowed step size increase factor
    MIN_S = 0.25  #: Minimum allowed step size reduction factor

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """Initialize an adaptive-step solver with validated configuration."""
        super().__init__(lin_op, nl_func, loglevel)
        self.config = config

        self.logger.debug(
            "Adaptive configuration: epsilon=%s, incr_f=%s, decr_f=%s, safety_f=%s",
            config.epsilon,
            config.incr_f,
            config.decr_f,
            config.safety_f,
        )

        self.__t0, self.__tf, self.__tc = 0, 0, 0
        self._accept = False

    # ------------------------------------------------------------------
    # Core Interface
    # ------------------------------------------------------------------
    @property
    def solver_type(self) -> SolverType:
        """
        Return the solver type for adaptive-step solvers.

        Returns
        -------
        SolverType
            Always returns ``SolverType.ADAPTIVE_STEP``.

        Examples
        --------
        >>> from rkstiff.etd35 import ETD35
        >>> solver = ETD35(lin_op, nl_func)
        >>> solver.solver_type == SolverType.ADAPTIVE_STEP
        True
        """
        return SolverType.ADAPTIVE_STEP

    def reset(self) -> None:
        """Reset solver and clear adaptive-step state."""
        self.logger.debug("Resetting adaptive solver state")
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
        Advance the solution one adaptive step.

        Must be implemented by subclasses to compute both the candidate
        next state and an estimate of the local truncation error.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Updated solution and local error estimate.
        """

    @abstractmethod
    def _q(self):
        """Return the method’s order (used in adaptive step size scaling)."""

    def step(self, u: np.ndarray, h_suggest: float) -> Tuple[np.ndarray, float, float]:
        """
        Perform one adaptive integration step.

        Attempts to advance the solution by time ``h_suggest`` and adjusts
        the step size automatically based on local error estimates.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h_suggest : float
            Suggested time step size.

        Returns
        -------
        unew : np.ndarray
            Updated solution vector after one accepted step.
        h : float
            Actual step size used.
        h_suggest : float
            Suggested step size for the next iteration.

        Raises
        ------
        MaxLoopsExceeded
            If too many attempts are made to find an acceptable step size.
        MinimumStepReached
            If the step size drops below the configured minimum ``minh``.

        Notes
        -----
        The algorithm follows this pattern:

        1. Try the proposed step.
        2. Estimate local error and compute scaling factor ``s``.
        3. If ``s < 1`` → reject step and reduce ``h``.
        4. If ``s ≥ 1`` → accept step and update ``h_suggest`` for next step.

        .. warning::
           Very small or divergent ``s`` values may indicate instability
           or excessively tight tolerances.
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
        r"""
        Compute the step size scaling factor :math:`s` for adaptive control.

        The scaling factor determines how the next step size should be
        adjusted based on the estimated local error:

        .. math::

            s = \text{safety\_f} \,
                \left( \frac{\varepsilon \, \|\mathbf{u}\|}
                            {\|\mathbf{e}\|} \right)^{1/q},

        where :math:`\varepsilon` is the tolerance, :math:`\mathbf{u}` is
        the current solution, :math:`\mathbf{e}` is the local error estimate,
        and :math:`q` is the method order returned by :meth:`_q`.

        To avoid instability from low-magnitude components, modes with
        relative amplitudes below ``config.adapt_cutoff`` are ignored.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector :math:`\mathbf{u}_n`.
        err : np.ndarray
            Local error estimate :math:`\mathbf{e}_n`.

        Returns
        -------
        float
            Step size scaling factor :math:`s`.
            - ``s > 1`` → step accepted (can increase step size)
            - ``s < 1`` → step rejected (reduce step size)

        Notes
        -----
        The new suggested step size is computed as :math:`h_{\text{new}} = s h`.
        """
        # Use adapt_cutoff to ignore small modes/values in the computation of the step size
        magu = np.abs(u)
        idx = magu / magu.max() > self.config.adapt_cutoff
        tol = self.config.epsilon * np.linalg.norm(u[idx])
        s = self.config.safety_f * np.power(tol / np.linalg.norm(err[idx]), 1.0 / self._q())
        return s

    def _reject_step_size(self, s: float, h: float) -> float:
        r"""
        Reduce the time step after a rejected attempt.

        Called when the estimated local error exceeds the tolerance.
        The method scales the current step size :math:`h` by the factor
        :math:`s`, clamped within stability limits.

        .. math::

            h_{\text{new}} = \max(s, s_{\min}) \;
                            \min(s, \text{decr\_f}) \; h

        Parameters
        ----------
        s : float
            Step size scaling factor computed by :meth:`_compute_s`.
        h : float
            Current step size before rejection.

        Returns
        -------
        float
            Reduced step size for retrying the integration.

        Notes
        -----
        - Enforces minimum allowed factor ``MIN_S`` to prevent collapse.
        - Logs detailed information if ``s`` is NaN or infinite.
        - Sets the acceptance flag ``_accept`` to ``False``.

        .. warning::
        Frequent rejections may indicate an unstable problem or
        overly strict tolerance.
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
        r"""
        Update the suggested step size after a successful integration step.

        When the estimated local error is within tolerance,
        the step is accepted, and the next step size is increased
        according to the scaling factor :math:`s`.

        .. math::

            h_{\text{next}} =
            \begin{cases}
                \min(s, s_{\max}) \, h, & \text{if } s > \text{incr\_f} \\
                h, & \text{otherwise.}
            \end{cases}

        Parameters
        ----------
        s : float
            Step size scaling factor computed by :meth:`_compute_s`.
        h : float
            Step size used for the accepted step.

        Returns
        -------
        float
            Suggested step size for the next step.

        Notes
        -----
        - Sets the acceptance flag ``_accept`` to ``True``.
        - Caps increases using ``MAX_S`` and ``config.incr_f`` to avoid
        overly aggressive growth.
        - Designed to ensure smooth variation of step size.
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
        r"""
        Integrate the system from :math:`t_0` to :math:`t_f` using adaptive time steps.

        Repeatedly applies :meth:`step` to propagate the solution forward
        while dynamically adjusting the time step size based on local
        error estimates.

        Parameters
        ----------
        u : np.ndarray
            Initial solution vector at :math:`t_0`.
        t0 : float
            Initial time.
        tf : float
            Final time.
        h_init : float, optional
            Initial step size. Defaults to ``(tf - t0) / 100`` if not provided.
        store_data : bool, default=True
            Whether to store intermediate results in :attr:`t` and :attr:`u`.
        store_freq : int, default=1
            Frequency of data storage; store every ``store_freq`` accepted steps.

        Returns
        -------
        np.ndarray
            Final solution at :math:`t = t_f`.

        Notes
        -----

        - The evolution proceeds until :math:`t \geq t_f`, automatically adjusting step sizes as needed.
        - Stored data is accessible via :attr:`t` and :attr:`u`.

        Example
        -------
        >>> solver = ETD35(lin_op, nl_func)
        >>> u_final = solver.evolve(u0, t0=0.0, tf=10.0)
        >>> solver.t[-1], np.linalg.norm(solver.u[-1])
        (10.0, 0.0134)
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
