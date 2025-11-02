r"""IF4 solver module
--------------------

Implements a fourth-order Integrating Factor (IF4) solver for stiff ODEs
of the form:

.. math::

    \frac{d\mathbf{u}}{dt} = L \mathbf{u} + N(\mathbf{u})

where :math:`L` is the linear operator and :math:`N(\mathbf{u})`
is the nonlinear term.

This module provides two internal strategies:
- :class:`_IF4Diagonal` — for diagonal (elementwise) linear operators
- :class:`_IF4NonDiagonal` — for full (matrix) linear operators

and a public solver class :class:`IF4` that wraps both and provides
a consistent constant-step interface.
"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .solvercs import StiffSolverCS


class _IF4Diagonal:  # pylint: disable=R0903
    """
    Strategy for IF4 solver with diagonal linear operator.

    Parameters
    ----------
    lin_op : np.ndarray
        Diagonal linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
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
        Update coefficients if step size h changed.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self.lin_op
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize N1 before first updateStage call.

        Parameters
        ----------
        u : np.ndarray
            State vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Perform one Runge-Kutta step.

        Parameters
        ----------
        u : np.ndarray
            State vector.
        h : float
            Step size.

        Returns
        -------
        np.ndarray
            Updated state vector.
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
        self._NL1 = self.nl_func(self._k)  # FSAL principle
        return self._k


class _IF4NonDiagonal:  # pylint: disable=R0903
    r"""
    Strategy for IF4 solver with non-diagonal linear operator.

    Uses full matrix exponentials computed via :func:`scipy.linalg.expm`.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`L` (square matrix).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`N(\mathbf{u})`.
    """

    def __init__(self, lin_op, nl_func):
        """Initialize IF4 non-diagonal strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        r"""
        Update matrix exponentials for current step size :math:`h`.

        Parameters
        ----------
        h : float
            Time step size.
        """
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        r"""
        Initialize :math:`N_1 = N(\mathbf{u}_n)`.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Perform one Runge–Kutta step using the matrix formulation.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Step size.

        Returns
        -------
        np.ndarray
            Updated state vector :math:`\mathbf{u}_{n+1}`.
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
        self._NL1 = self.nl_func(self._k)  # FSAL principle
        return self._k


class IF4(StiffSolverCS):
    r"""
    Fourth-order Integrating Factor (IF4) constant-step solver.

    This solver advances stiff ODEs of the form:

    .. math::

        \frac{d\mathbf{u}}{dt} = L \mathbf{u} + N(\mathbf{u})

    using a fourth-order Runge–Kutta integrating factor method.
    It supports both diagonal and non-diagonal linear operators.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator (matrix or diagonal).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`N(\mathbf{u})`.
    loglevel : str or int, optional
        Logging level (default ``"WARNING"``).

    Notes
    -----
    This method is based on the integrating factor RK4 scheme:

    .. math::

        \mathbf{u}_{n+1} = e^{hL} \mathbf{u}_n
        + h \sum_{i=1}^4 b_i e^{c_i hL} N_i

    where the coefficients :math:`b_i` and :math:`c_i` correspond
    to the classical fourth-order Runge–Kutta tableau.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize IF4 solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator (matrix or diagonal).
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        loglevel : str or int, optional
            Logging level.
        """
        super().__init__(lin_op, nl_func, loglevel)
        self._method = Union[_IF4Diagonal, _IF4NonDiagonal]
        if self._diag:
            self._method = _IF4Diagonal(lin_op, nl_func)
        else:
            self._method = _IF4NonDiagonal(lin_op, nl_func)
        self.__n1_init = False
        self._h_coeff = None

    def _reset(self) -> None:
        """Reset solver to its initial state."""
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h: float) -> None:
        """Update exponential coefficients if step size :math:`h` changed."""
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("IF4 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""Compute :math:`\mathbf{u}_{n+1}` from :math:`\mathbf{u}_n` in one step."""
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h)
