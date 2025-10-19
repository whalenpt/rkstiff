""" "IF4 solver module"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.solver import StiffSolverCS


class _IF4Diagonal:  # pylint: disable=R0903
    """IF4 diagonal system strategy for IF4 solver"""

    def __init__(self, lin_op, nl_func):
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h):
        """Update coefficients if step size h changed"""
        z = h * self.lin_op
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u):
        """Need to initialize N1 before first updateStage call"""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u, h):
        """One RK step"""
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
    """IF4 non-diagonal system strategy for IF4 solver"""

    def __init__(self, lin_op, nl_func):
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h):
        """Update coefficients if step size h changed"""
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

    def n1_init(self, u):
        """Need to initialize N1 before first updateStage call"""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u, h):
        """One RK step"""
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
    """
    Integrating factor constant step solver of 4th order

    ATTRIBUTES
    __________
    lin_op : np.array
    nl_func : function
    t : time-array stored with evolve function call
    u : output-array stored with evolve function call
    logs : array of info stored related to the solver

    StiffSolverAS Parameters (see StiffSolverAS class in solver module)
    ________________________
    epsilon : float
    incrF : float
    decrF : float
    safetyF : float
    adapt_cutoff : float
    minh : float
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ):
        """
        INPUTS
        ______

        lin_op : np.array
            Linear operator (L) in the system dtU = LU + NL(U). Can be either a 2D numpy array (matrix)
            or a 1D array (diagonal system). L can be either real-valued or complex-valued.

        nl_func : function
            Nonlinear function (NL(U)) in the system dtU = LU + NL(U). Can be a complex or real-valued function.

        """

        super().__init__(lin_op, nl_func, loglevel)
        self._method = Union[_IF4Diagonal, _IF4NonDiagonal]
        if self._diag:
            self._method = _IF4Diagonal(lin_op, nl_func)
        else:
            self._method = _IF4NonDiagonal(lin_op, nl_func)
        self.__n1_init = False
        self._h_coeff = None

    def _reset(self):
        """Resets solver to its initial state"""
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h):
        """Update coefficients if step size h changed"""
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("IF4 coefficients updated for step size h=%s", h)

    def _update_stages(self, u, h):
        """compute_s u_{n+1} from u_{n} through one RK passthrough"""
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h)
