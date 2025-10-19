"""rkstiff.etd34: Exponential time-differencing adaptive step solver of 4th order."""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.solver import SolverConfig
from rkstiff.etd import ETDAS, ETDConfig, phi1, phi2, phi3


class _Etd34Diagonal:  # pylint: disable=too-few-public-methods
    """ETD34 diagonal system strategy for ETD34 solver."""
    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """Initialize ETD34 diagonal system strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
        ) = [np.zeros(n, dtype=np.complex128) for _ in range(8)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update internal ETD coefficients for step size h."""
        z = h * self.lin_op
        self._update_coeffs_diagonal(h, z)

    def _update_coeffs_diagonal(self, h: float, z: np.ndarray) -> None:
        """Compute elementwise ETD coefficients for diagonal z = h*L."""
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

        smallmode_idx = np.abs(z) < self.etd_config.modecutoff
        zb = z[~smallmode_idx]  # z big
        # compute big mode coeffs
        phi1_12 = h * phi1(zb / 2)
        phi2_12 = h * phi2(zb / 2)
        phi1_1 = h * phi1(zb)
        phi2_1 = h * phi2(zb)
        phi3_1 = h * phi3(zb)

        self._a21[~smallmode_idx] = 0.5 * phi1_12
        self._a31[~smallmode_idx] = 0.5 * (phi1_12 - phi2_12)
        self._a32[~smallmode_idx] = 0.5 * phi2_12
        self._a41[~smallmode_idx] = phi1_1 - phi2_1
        self._a43[~smallmode_idx] = phi2_1
        self._a51[~smallmode_idx] = phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1
        self._a52[~smallmode_idx] = phi2_1 - (2.0 / 3) * phi3_1
        self._a54[~smallmode_idx] = -(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1

        # compute small mode coeffs
        zs = z[smallmode_idx]  # z small
        r = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        phi1_12 = h * np.sum(phi1(Z / 2), axis=1) / self.etd_config.contour_points
        phi2_12 = h * np.sum(phi2(Z / 2), axis=1) / self.etd_config.contour_points
        phi1_1 = h * np.sum(phi1(Z), axis=1) / self.etd_config.contour_points
        phi2_1 = h * np.sum(phi2(Z), axis=1) / self.etd_config.contour_points
        phi3_1 = h * np.sum(phi3(Z), axis=1) / self.etd_config.contour_points

        self._a21[smallmode_idx] = 0.5 * phi1_12
        self._a31[smallmode_idx] = 0.5 * (phi1_12 - phi2_12)
        self._a32[smallmode_idx] = 0.5 * phi2_12
        self._a41[smallmode_idx] = phi1_1 - phi2_1
        self._a43[smallmode_idx] = phi2_1
        self._a51[smallmode_idx] = phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1
        self._a52[smallmode_idx] = phi2_1 - (2.0 / 3) * phi3_1
        self._a54[smallmode_idx] = -(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize stored nonlinear evaluation N1 = nl_func(u)."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform the RK stage updates and return (k, error_estimate).
        """
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()
        # If not accept, then step failed, reuse previously computed N1
        self._k = self._EL2 * u + self._a21 * self._NL1
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2 * u + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL * u + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL * u + self._a51 * self._NL1 + self._a52 * (self._NL2 + self._NL3) + self._a54 * self._NL4
        self._NL5 = self.nl_func(self._k)
        self._err = self._a54 * (self._NL4 - self._NL5)
        return self._k, self._err


class _Etd34Diagonalized(_Etd34Diagonal):
    """
    ETD34 non-diagonal system with eigenvector diagonalization strategy for ETD34 solver
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig):
        """Initialize diagonalized strategy; computes eigen-decomposition of lin_op."""
        super().__init__(lin_op, nl_func, etd_config)
        if len(lin_op.shape) == 1:
            raise ValueError("cannot diagonalize a 1D system")
        lin_op_cond = np.linalg.cond(lin_op)
        if lin_op_cond > 1e16:
            raise ValueError("cannot diagonalize a non-invertible linear operator L")
        if lin_op_cond > 1000:
            # Provide a friendly, single-line warning.
            print(
                f"Warning: linear matrix array has a large condition number of "
                f"{lin_op_cond:.2f}, method may be unstable"
            )
        self._eig_vals, self._S = np.linalg.eig(lin_op)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(lin_op.shape[0])

    def update_coeffs(self, h: float) -> None:
        """Update coefficients for diagonalized eigenvalues."""
        z = h * self._eig_vals
        self._update_coeffs_diagonal(h, z)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize stored nonlinear evaluation and transformed state v = S^{-1} u."""
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform stage updates in the diagonalized basis and return (u_next, err)."""
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()
            self._v = self._Sinv.dot(u)

        self._k = self._EL2 * self._v + self._a21 * self._NL1
        self._NL2 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL2 * self._v + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL * self._v + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = (
            self._EL * self._v + self._a51 * self._NL1 + self._a52 * (self._NL2 + self._NL3) + self._a54 * self._NL4
        )
        self._NL5 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._err = self._a54 * (self._NL4 - self._NL5)
        return self._S.dot(self._k), self._err


class _Etd34NonDiagonal:
    """
    ETD34 non-diagonal system strategy for ETD34 solver
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig):
        """Initialize the non-diagonal strategy."""
        self.lin_op = lin_op
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(2)]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
        ) = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(8)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update matrix-valued ETD coefficients for step size ``h``."""
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

        contour_points = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )

        phi1_12, phi2_12, phi1_1, phi2_1, phi3_1 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        for point in contour_points:
            Q = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z)
            Q2 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 2)
            phi1_12 += point * phi1(point) * Q2 / self.etd_config.contour_points
            phi2_12 += point * phi2(point) * Q2 / self.etd_config.contour_points
            phi1_1 += point * phi1(point) * Q / self.etd_config.contour_points
            phi2_1 += point * phi2(point) * Q / self.etd_config.contour_points
            phi3_1 += point * phi3(point) * Q / self.etd_config.contour_points

        self._a21 = 0.5 * h * phi1_12
        self._a31 = 0.5 * h * (phi1_12 - phi2_12)
        self._a32 = 0.5 * h * phi2_12
        self._a41 = h * (phi1_1 - phi2_1)
        self._a43 = h * phi2_1
        self._a51 = h * (phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1)
        self._a52 = h * (phi2_1 - (2.0 / 3) * phi3_1)
        self._a54 = h * (-(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1)

    def n1_init(self, u: np.ndarray) -> None:
        """Initialize stored nonlinear evaluation N1 = nl_func(u)."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform stage updates for the full matrix strategy and return (k, err)."""
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()

        self._k = self._EL2.dot(u) + self._a21.dot(self._NL1)
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2.dot(u) + self._a31.dot(self._NL1) + self._a32.dot(self._NL2)
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL.dot(u) + self._a41.dot(self._NL1) + self._a43.dot(self._NL3)
        self._NL4 = self.nl_func(self._k)
        self._k = (
            self._EL.dot(u)
            + self._a51.dot(self._NL1)
            + self._a52.dot(self._NL2 + self._NL3)
            + self._a54.dot(self._NL4)
        )
        self._NL5 = self.nl_func(self._k)
        self._err = self._a54.dot(self._NL4 - self._NL5)
        return self._k, self._err


class ETD34(ETDAS):
    """
    Fourth-order Exponential Time Differencing (ETD) integrator with adaptive stepping.

    This class implements the ETD(3,4) scheme, a fourth-order exponential integrator
    that adaptively adjusts the time step based on embedded error estimation. It wraps
    the lower-level per-strategy implementations (diagonal, diagonalized, and non-diagonal)
    and leverages the adaptive controller provided by the :class:`ETDAS` base class.

    The governing equation is assumed to be of the form:
        dU/dt = L·U + N(U)

    where `L` is a linear operator and `N(U)` is a nonlinear function.

    Attributes
    ----------
    lin_op : np.ndarray
        Linear operator `L` in the system `dU/dt = L·U + N(U)`. Can be either:
        - A 2D matrix (for full linear operators)
        - A 1D array (for diagonal systems)
        `L` may be real-valued or complex-valued.

    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function `N(U)` in the system `dU/dt = L·U + N(U)`.
        Can be real-valued or complex-valued.

    _method : Union[_Etd34Diagonal, _Etd34Diagonalized, _Etd34NonDiagonal]
        Internal method implementation selected based on the form of `L`.

    Notes
    -----
    Configuration parameters for contour integration, adaptivity, and safety factors
    are inherited from the :class:`ETDAS` and :class:`StiffSolverAS` base classes.
    """
    _method: Union[_Etd34Diagonal, _Etd34Diagonalized, _Etd34NonDiagonal]

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        etd_config: ETDConfig = ETDConfig(),
        diagonalize: bool = False,
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ):
        """
        Initialize the ETD(3,4) solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator `L` in the system `dU/dt = L·U + N(U)`. May be
            either a 2D NumPy matrix or a 1D array representing a diagonal system.
            Supports both real and complex-valued operators.

        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function `N(U)` in the system `dU/dt = L·U + N(U)`.

        config : SolverConfig, optional
            General solver configuration controlling adaptivity thresholds,
            safety factors, and other integration parameters.

        etd_config : ETDConfig, optional
            Configuration for ETD-specific parameters, such as contour integration
            settings and spectral radius estimation.

        diagonalize : bool, optional
            If True, the solver diagonalizes the linear operator `L` before integration.
            This can improve performance for certain non-diagonalizable but sparse systems.

        Notes
        -----
        The following parameters are inherited from parent classes:
        - From :class:`ETDAS`: `modecutoff`, `contour_points`, `contour_radius`
        - From :class:`StiffSolverAS`: `epsilon`, `incrF`, `decrF`, `safetyF`,
          `adapt_cutoff`, and `minh`
        """
        super().__init__(lin_op, nl_func, config=config, etd_config=etd_config, loglevel=loglevel)
        if self._diag:
            self._method = _Etd34Diagonal(lin_op, nl_func, self.etd_config)
        else:
            if diagonalize:
                self._method = _Etd34Diagonalized(lin_op, nl_func, etd_config)
            else:
                self._method = _Etd34NonDiagonal(lin_op, nl_func, self.etd_config)
        self.__n1_init = False
        self._accept = False

    def _reset(self) -> None:
        """
        Reset internal solver state.

        This method resets adaptive-step control flags and cached coefficients.
        It is called when the solver starts or restarts an integration sequence.
        """
        # Resets solver to its initial state
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD coefficients for a new time step.

        Parameters
        ----------
        h : float
            Current time step size.

        Notes
        -----
        The coefficient update is skipped if the time step `h` has not changed
        since the last update.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD34 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute u_{n+1} (and an error estimate) from u_n through one RK pass."""
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, self._accept)

    def _q(self) -> int:
        """
        Return the solver order used for adaptive step control.

        Returns
        -------
        int
            Effective order (4 for ETD(3,4)), used by the adaptive controller
            for error-based time step adjustments.
        """
        return 4
