"""IF34: Integrating Factor 4(3) adaptive step solver"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.solver import StiffSolverAS, SolverConfig


class _If34Diagonal:  # pylint: disable=too-few-public-methods
    """
    IF34 diagonal system strategy for IF34 solver.

    Optimized implementation for diagonal linear operators using element-wise operations.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Initialize IF34 diagonal system strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update coefficients based on step size h."""
        z = h * self.lin_op
        self._update_coeffs_diagonal(z)

    def _update_coeffs_diagonal(self, z: np.ndarray) -> None:
        """Compute element-wise coefficients for diagonal z = h*L."""
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize first nonlinear evaluation NL1 = nl_func(u_n)"""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Update stages and return u_{n+1} and error estimate.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time step size.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Next state vector and error estimate.
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


class _If34Diagonalized(_If34Diagonal):
    """
    IF34 non-diagonal system with eigenvector diagonalization strategy.

    Uses eigenvalue decomposition to transform the system into diagonal form.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Initialize IF34 diagonalized system strategy"""
        super().__init__(lin_op, nl_func)
        if len(lin_op.shape) == 1:
            raise ValueError("Cannot diagonalize a 1D system")
        lin_op_cond = np.linalg.cond(lin_op)
        if lin_op_cond > 1e16:
            raise ValueError("Cannot diagonalize a non-invertible linear operator L")
        if lin_op_cond > 1000:
            print(
                f"Warning: linear matrix array has a large condition number of "
                f"{lin_op_cond:.2f}, method may be unstable"
            )
        self._eig_vals, self._S = np.linalg.eig(lin_op)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(lin_op.shape[0])

    def update_coeffs(self, h: float) -> None:
        """Update coefficients using eigenvalues."""
        z = h * self._eig_vals
        self._update_coeffs_diagonal(z)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize first nonlinear evaluation in transformed basis."""
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u, h, accept):
        """Update stages and return u_{n+1} and error estimate"""
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
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


class _If34NonDiagonal:  # pylint: disable=too-few-public-methods
    """
    IF34 non-diagonal system strategy using full matrix operations.

    Uses matrix exponentials for general (non-diagonalizable) linear operators.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Initialize IF34 non-diagonal system strategy"""
        self.lin_op = lin_op
        self.nl_func = nl_func

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update matrix exponential coefficients based on step size h."""
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize first nonlinear evaluation NL1 = nl_func(u_n)."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, h: float, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Update stages and return u_{n+1} and error estimate"""
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
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


class IF34(StiffSolverAS):
    """
    Fourth-order Integrating Factor solver with adaptive stepping.

    Implements the IF(3,4) scheme with an embedded third-order method for
    error estimation and adaptive time step control. Suitable for stiff
    systems of the form dU/dt = L*U + NL(U).

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator L in the system dU/dt = L*U + NL(U).
        Can be 1D (diagonal) or 2D (full matrix).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function nl_func(U).
    config : SolverConfig, optional
        Solver configuration for adaptive stepping parameters.
    diagonalize : bool, optional
        If True, diagonalize the linear operator via eigenvalue decomposition.

    Attributes
    ----------
    t : np.ndarray
        Time values from most recent call to evolve().
    u : np.ndarray
        Solution array from most recent call to evolve().
    logs : list
        Log messages recording solver operations.

    Notes
    -----
    Adaptive stepping parameters (epsilon, safety factors, etc.) are
    inherited from the StiffSolverAS base class via the config parameter.
    """

    _method: Union[_If34Diagonal, _If34Diagonalized, _If34NonDiagonal]

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        diagonalize: bool = False,
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """Initialize the IF34 adaptive solver."""
        super().__init__(lin_op, nl_func, config=config, loglevel=loglevel)
        self._method = Union[_If34Diagonal, _If34Diagonalized, _If34NonDiagonal]
        if self._diag:
            self._method = _If34Diagonal(lin_op, nl_func)
        else:
            if diagonalize:
                self._method = _If34Diagonalized(lin_op, nl_func)
            else:
                self._method = _If34NonDiagonal(lin_op, nl_func)
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _reset(self) -> None:
        """Reset solver to its initial state."""
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """Update coefficients if step size h changed."""
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("IF34 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute u_{n+1} from u_n through one RK pass."""
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h, self._accept)

    def _q(self) -> int:
        """Return order for computing suggested step size (embedded order + 1)."""
        return 4
