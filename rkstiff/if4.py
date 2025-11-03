r"""
Constant-Step Fourth-Order Integrating Factor Integrator
========================================================

Implements a fourth-order Integrating Factor (IF4) solver for stiff
partial differential equations (PDEs) of the form:

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is the linear spatial operator and
:math:`\mathcal{N}(\mathbf{U})` is the nonlinear term.

After spatial discretization (e.g., using finite differences,
Fourier spectral methods, or finite elements), this PDE system
reduces to a stiff system of ordinary differential equations in time,
which the IF4 scheme integrates efficiently.

This module provides two internal solver strategies:

* :class:`_IF4Diagonal` — optimized for diagonal (spectral) operators.
* :class:`_IF4NonDiagonal` — general matrix-based implementation.

The public class :class:`IF4` wraps both strategies and provides a
consistent constant-step interface for time evolution.
"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .solvercs import BaseSolverCS


class _IF4Diagonal:
    r"""
    Internal strategy for IF4 with diagonal spatial operator.

    Used when the linear term :math:`\mathcal{L}` acts diagonally
    in the chosen spatial representation (e.g., Fourier spectral space).
    This allows for highly efficient elementwise updates.

    The semi-discrete system advanced is:

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    and each spatial mode evolves independently.

    Parameters
    ----------
    lin_op : np.ndarray
        1-D array representing the diagonal of :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear term :math:`\mathcal{N}(\mathbf{U})`.
    """

    def __init__(self, lin_op, nl_func):
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """
        Update exponential coefficients for step size :math:`h`.

        Parameters
        ----------
        h : float
            Time-step size.
        """
        z = h * self.lin_op
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        r"""
        Initialize first nonlinear term :math:`\mathcal{N}_1 = \mathcal{N}(\mathbf{U}_n)`.

        Parameters
        ----------
        u : np.ndarray
            Current state vector in spectral or physical space.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Perform one fourth-order integrating factor Runge–Kutta step.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time-step size.

        Returns
        -------
        np.ndarray
            Updated state vector :math:`\mathbf{U}_{n+1}`.
        """
        self._k = self._EL2 * u + h * self._EL2 * self._NL1 / 2.0
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2 * u + h * self._NL2 / 2.0
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL * u + h * self._EL2 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL * u + h * (
            self._EL * self._NL1 / 6.0 + self._EL2 * self._NL2 / 3.0 + self._EL2 * self._NL3 / 3.0 + self._NL4 / 6.0
        )
        self._NL1 = self.nl_func(self._k)  # FSAL (First Same As Last)
        return self._k


class _IF4NonDiagonal:
    r"""
    Internal strategy for IF4 with non-diagonal spatial operator.

    Uses full matrix exponentials computed via :func:`scipy.linalg.expm`
    to propagate the linear operator exactly at each time step.
    This general form applies when :math:`\mathcal{L}` cannot be
    diagonalized efficiently.

    Parameters
    ----------
    lin_op : np.ndarray
        Square matrix representation of :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear term :math:`\mathcal{N}(\mathbf{U})`.
    """

    def __init__(self, lin_op, nl_func):
        """Initialize non-diagonal IF4 strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        r"""
        Compute matrix exponentials :math:`e^{h\mathcal{L}}`
        and :math:`e^{(h/2)\mathcal{L}}` for the given step.

        Parameters
        ----------
        h : float
            Time-step size.
        """
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        r"""Initialize the first nonlinear evaluation :math:`\mathcal{N}_1 = \mathcal{N}(\mathbf{U}_n)`."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Advance the solution by one IF4 step using full matrix exponentials.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time-step size.

        Returns
        -------
        np.ndarray
            Updated solution :math:`\mathbf{U}_{n+1}`.
        """
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
        self._NL1 = self.nl_func(self._k)
        return self._k


class IF4(BaseSolverCS):
    r"""
    Fourth-order Integrating Factor (IF4) constant-step solver for PDEs.

    Advances semi-discretized PDE systems of the form:

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` is the linear spatial operator and
    :math:`\mathcal{N}` the nonlinear operator.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}` (matrix or diagonal array).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    loglevel : str or int, optional
        Logging verbosity level (default ``"WARNING"``).

    Notes
    -----
    The IF4 method applies the classical fourth-order Runge–Kutta
    scheme in the exponential integrating factor framework:

    .. math::

        \mathbf{U}_{n+1}
            = e^{h\mathcal{L}}\mathbf{U}_n
            + h \sum_{i=1}^{4} b_i e^{c_i h \mathcal{L}} \mathcal{N}_i,

    where :math:`(b_i, c_i)` are the coefficients of the RK4 tableau.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """Initialize the IF4 solver and select diagonal or matrix strategy."""
        super().__init__(lin_op, nl_func, loglevel)
        self._method = Union[_IF4Diagonal, _IF4NonDiagonal]
        if self._diag:
            self._method = _IF4Diagonal(lin_op, nl_func)
        else:
            self._method = _IF4NonDiagonal(lin_op, nl_func)
        self.__n1_init = False
        self._h_coeff = None

    def _reset(self) -> None:
        """Reset solver to its initial (pre-evolution) state."""
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h: float) -> None:
        """Update exponential coefficients if step size :math:`h` changes."""
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("IF4 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""Advance :math:`\mathbf{U}_n` to :math:`\mathbf{U}_{n+1}` in one IF4 step."""
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h)
