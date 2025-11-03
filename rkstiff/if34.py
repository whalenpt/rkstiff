r"""
Adaptive-Step Fourth Order (Third Order Embedding) Integrating Factor Integrator
================================================================================

Adaptive Integrating Factor 4(3) solver.

Implements the **IF(3,4)** exponential Runge–Kutta scheme with an embedded
third-order method for local error estimation and adaptive step control.
It solves stiff semi-linear systems of the form

.. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is the linear (stiff) operator and
:math:`\mathcal{N}` the nonlinear term.

The IF(3,4) method integrates this system in the exponential form

.. math::

    \mathbf{U}_{n+1} = e^{h\mathcal{L}}\mathbf{U}_n
        + h \sum_{i=1}^{s} b_i \,
        e^{(1-c_i)h\mathcal{L}} \, \mathcal{N}(\mathbf{U}_i),

where the intermediate stages :math:`\mathbf{U}_i` are computed using
exponential operators and the nonlinear evaluations.

The embedded third-order estimate is used to compute adaptive step sizes
according to local error tolerances.

References
----------
P. Whalen, M. Brio, and J. V. Moloney,
*Exponential time-differencing with embedded Runge-Kutta adaptive step control*,
J. Comput. Phys. **280**, 579-601 (2015).
"""

import logging
from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .solveras import SolverConfig, BaseSolverAS


# ======================================================================
# Diagonal operator strategy
# ======================================================================
class _If34Diagonal:
    r"""
    IF(3,4) diagonal strategy.

    Optimized implementation for diagonal linear operators using
    element-wise exponentials. Solves

    .. math::
        \frac{d\mathbf{U}}{dt} = \Lambda \mathbf{U} + \mathcal{N}(\mathbf{U}),

    where :math:`\Lambda` is diagonal.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """Initialize the diagonal IF(3,4) system strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func
        self.logger = logger

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        r"""
        Update exponential coefficients for the given step size.

        .. math::

            z = h \Lambda, \qquad
            E_L = e^{z}, \qquad
            E_{L/2} = e^{z/2}.
        """
        z = h * self.lin_op
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        r"""Initialize the first nonlinear evaluation :math:`\mathcal{N}_1 = \mathcal{N}(\mathbf{u}_n)`."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the IF(3,4) Runge–Kutta stages and error estimate.

        Parameters
        ----------
        u : np.ndarray
            Current state :math:`\mathbf{u}_n`.
        h : float
            Step size.
        accept : bool
            Whether the previous step was accepted (FSAL reuse).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Next state and local error estimate.
        """
        if accept:
            self._NL1 = self._NL5.copy()

        self._k = self._EL2 * u + h * self._EL2 * self._NL1 / 2.0
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2 * u + h * self._NL2 / 2.0
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL * u + h * self._EL2 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL * u + h * (
            self._EL * self._NL1 / 6.0 + self._EL2 * self._NL2 / 3.0 + self._EL2 * self._NL3 / 3.0 + self._NL4 / 6.0
        )
        self._NL5 = self.nl_func(self._k)
        self._err = h * (self._NL4 - self._NL5) / 6.0
        return self._k, self._err


# ======================================================================
# Diagonalized via eigen-decomposition
# ======================================================================
class _If34Diagonalized(_If34Diagonal):
    r"""
    IF(3,4) strategy for diagonalizable linear systems.

    Performs eigenvalue decomposition

    .. math::
        \mathcal{L} = S \Lambda S^{-1},

    then evolves the system in the diagonal eigenbasis for efficiency.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """Initialize eigen-basis transformation for the linear operator."""
        super().__init__(lin_op, nl_func, logger)
        if len(lin_op.shape) == 1:
            raise ValueError("Cannot diagonalize a 1D system")
        cond = np.linalg.cond(lin_op)
        if cond > 1e16:
            raise ValueError("Linear operator is near-singular and cannot be diagonalized")
        if cond > 1000:
            self.logger.warning(f"High condition number ({cond:.2e}); diagonalization may be unstable")
        self._eig_vals, self._S = np.linalg.eig(lin_op)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(lin_op.shape[0])

    def update_coeffs(self, h: float) -> None:
        """Update exponentials based on eigenvalues."""
        z = h * self._eig_vals
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize in the diagonalized (eigen) basis."""
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u: np.ndarray, h: float, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Compute stages and return next state + error in transformed basis."""
        if accept:
            self._NL1 = self._NL5.copy()
            self._v = self._Sinv.dot(u)

        self._k = self._EL2 * self._v + h * self._EL2 * self._NL1 / 2.0
        self._NL2 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL2 * self._v + h * self._NL2 / 2.0
        self._NL3 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL * self._v + h * self._EL2 * self._NL3
        self._NL4 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL * self._v + h * (
            self._EL * self._NL1 / 6.0 + self._EL2 * self._NL2 / 3.0 + self._EL2 * self._NL3 / 3.0 + self._NL4 / 6.0
        )
        self._NL5 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._err = h * (self._NL4 - self._NL5) / 6.0
        return self._S.dot(self._k), self._err


# ======================================================================
# Full matrix exponential (non-diagonal)
# ======================================================================
class _If34NonDiagonal:
    r"""
    IF(3,4) strategy for full (non-diagonalizable) linear operators.

    Uses direct matrix exponentials:

    .. math::

        E_L = e^{h\mathcal{L}}, \qquad
        E_{L/2} = e^{h\mathcal{L}/2}.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Initialize the general IF(3,4) solver strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update matrix exponentials for current step size."""
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize nonlinear evaluation."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Compute IF(3,4) stages and error using full matrix operations."""
        if accept:
            self._NL1 = self._NL5.copy()

        self._k = self._EL2.dot(u) + h * self._EL2.dot(self._NL1 / 2.0)
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2.dot(u) + h * self._NL2 / 2.0
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL.dot(u) + h * self._EL2.dot(self._NL3)
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL.dot(u) + h * (
            self._EL.dot(self._NL1 / 6.0)
            + self._EL2.dot(self._NL2 / 3.0)
            + self._EL2.dot(self._NL3 / 3.0)
            + self._NL4 / 6.0
        )
        self._NL5 = self.nl_func(self._k)
        self._err = h * (self._NL4 - self._NL5) / 6.0
        return self._k, self._err


# ======================================================================
# Public adaptive solver
# ======================================================================
class IF34(BaseSolverAS):
    r"""
    Adaptive Integrating-Factor 4(3) solver.

    Fourth-order integrating factor Runge–Kutta scheme with an embedded
    third-order pair for local error control.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    config : SolverConfig, optional
        Adaptive step configuration.
    diagonalize : bool, default=False
        Attempt eigenvalue diagonalization if linear operator is 2D.
    loglevel : str or int, default='WARNING'
        Logging verbosity.

    Notes
    -----
    - Implements **FSAL** (First Same As Last) reuse for efficiency.
    - Coefficients are recomputed only when step size changes.
    - Supports diagonal, diagonalizable, and full-matrix systems.
    """

    _method: Union[_If34Diagonal, _If34Diagonalized, _If34NonDiagonal]

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        diagonalize: bool = False,
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """Initialize the IF(3,4) adaptive solver."""
        super().__init__(lin_op, nl_func, config=config, loglevel=loglevel)
        self._method = (
            _If34Diagonal(lin_op, nl_func, self.logger)
            if self._diag
            else (_If34Diagonalized(lin_op, nl_func, self.logger) if diagonalize else _If34NonDiagonal(lin_op, nl_func))
        )
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _reset(self) -> None:
        """Reset solver state (reinitializes stage storage and flags)."""
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """Recompute coefficients if step size changed."""
        if h != self._h_coeff:
            self._h_coeff = h
            self._method.update_coeffs(h)
            self.logger.debug("IF34 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Perform one adaptive IF(3,4) integration step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Next state and local error vector :math:`\mathbf{e}_n`.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h, self._accept)

    def _q(self) -> int:
        """Order of method used for adaptive control (4)."""
        return 4
