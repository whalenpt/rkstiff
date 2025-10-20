"""ETD35 Exponential Time Differencing Adaptive Step Solver.

Implements the ETD35 (5th order, 3rd order embedded) adaptive step exponential
time-differencing algorithm for stiff ODE systems. Supports diagonal,
diagonalized, and full non-diagonal systems.
"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.etd import ETDAS, ETDConfig, phi1, phi2, phi3
from rkstiff.solver import SolverConfig


class _Etd35Diagonal:
    """ETD35 diagonal system strategy for ETD35 solver."""

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """Initialize ETD35 diagonal solver state."""
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL14, self._EL12, self._EL34, self._EL = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
            self._a61,
            self._a62,
            self._a63,
            self._a64,
            self._a65,
            self._a71,
            self._a73,
            self._a74,
            self._a75,
            self._a76,
        ) = [np.zeros(n, dtype=np.complex128) for _ in range(18)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """Update ETD coefficients for given step size `h`."""
        z = h * self.lin_op
        self._update_coeffs_diagonal(h, z)

    def _update_coeffs_diagonal(self, h: float, z: np.ndarray) -> None:
        """Internal coefficient update for diagonal systems."""
        self._EL14 = np.exp(z / 4.0)
        self._EL12 = np.exp(z / 2.0)
        self._EL34 = np.exp(3.0 * z / 4.0)
        self._EL = np.exp(z)

        smallmode_idx = np.abs(z) < self.etd_config.modecutoff

        self._update_large_mode_coeffs(h, z, smallmode_idx)
        self._update_small_mode_coeffs(h, z, smallmode_idx)

    def _update_large_mode_coeffs(self, h: float, z: np.ndarray, smallmode_idx: np.ndarray) -> None:
        """Compute ETD35 coefficients for modes with |z| >= cutoff."""
        idx = ~smallmode_idx

        # Early return if no large modes
        if not np.any(idx):
            return

        zb = z[idx]  # z big - extract large modes

        # Compute phi functions for large modes
        phi1_14, phi2_14 = h * phi1(zb / 4), h * phi2(zb / 4)
        phi1_12, phi2_12 = h * phi1(zb / 2), h * phi2(zb / 2)
        phi1_34, phi2_34 = h * phi1(3 * zb / 4), h * phi2(3 * zb / 4)
        phi1_1, phi2_1, phi3_1 = h * phi1(zb), h * phi2(zb), h * phi3(zb)

        # Assign directly to the indexed locations
        # phi1_14 etc. are already filtered, so we assign to self._a21[idx]

        self._a21[idx] = phi1_14 / 4.0
        self._a31[idx] = (phi1_14 - phi2_14 / 2.0) / 4.0
        self._a32[idx] = phi2_14 / 8.0
        self._a41[idx] = (phi1_12 - phi2_12) / 2.0
        self._a43[idx] = phi2_12 / 2.0
        self._a51[idx] = 3.0 * (phi1_34 - 3.0 * phi2_34 / 4.0) / 4.0
        self._a52[idx] = -3 * phi1_34 / 8.0
        self._a54[idx] = 9 * phi2_34 / 16.0
        self._a61[idx] = (-77 * phi1_1 + 59 * phi2_1) / 42.0
        self._a62[idx] = 8 * phi1_1 / 7.0
        self._a63[idx] = (111 * phi1_1 - 87 * phi2_1) / 28.0
        self._a65[idx] = (-47 * phi1_1 + 143 * phi2_1) / 84.0
        self._a71[idx] = 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        # Paper has error in a73/b3 phi2 coefficient (states this is 497 but it is actually 467)
        self._a73[idx] = (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._a74[idx] = 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._a75[idx] = (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._a76[idx] = (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

    def _update_small_mode_coeffs(  # pylint: disable=too-many-locals
        self, h: float, z: np.ndarray, smallmode_idx: np.ndarray
    ) -> None:
        """Compute ETD35 coefficients for modes with |z| < cutoff using contour integration."""
        if not np.any(smallmode_idx):
            return

        zs = z[smallmode_idx]
        npts = self.etd_config.contour_points
        r = self.etd_config.contour_radius * np.exp(2j * np.pi * np.arange(0.5, npts) / npts)
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        def avg_phi(func, factor=1):
            return h * np.sum(func(factor * Z), axis=1) / npts

        phi1_14, phi2_14 = avg_phi(phi1, 1 / 4), avg_phi(phi2, 1 / 4)
        phi1_12, phi2_12 = avg_phi(phi1, 1 / 2), avg_phi(phi2, 1 / 2)
        phi1_34, phi2_34 = avg_phi(phi1, 3 / 4), avg_phi(phi2, 3 / 4)
        phi1_1, phi2_1, phi3_1 = avg_phi(phi1), avg_phi(phi2), avg_phi(phi3)

        idx = smallmode_idx
        self._a21[idx] = phi1_14 / 4.0
        self._a31[idx] = (phi1_14 - phi2_14 / 2.0) / 4.0
        self._a32[idx] = phi2_14 / 8.0
        self._a41[idx] = (phi1_12 - phi2_12) / 2.0
        self._a43[idx] = phi2_12 / 2.0
        self._a51[idx] = 3.0 * (phi1_34 - 3.0 * phi2_34 / 4.0) / 4.0
        self._a52[idx] = -3 * phi1_34 / 8.0
        self._a54[idx] = 9 * phi2_34 / 16.0
        self._a61[idx] = (-77 * phi1_1 + 59 * phi2_1) / 42.0
        self._a62[idx] = 8 * phi1_1 / 7.0
        self._a63[idx] = (111 * phi1_1 - 87 * phi2_1) / 28.0
        self._a65[idx] = (-47 * phi1_1 + 143 * phi2_1) / 84.0
        self._a71[idx] = 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        self._a73[idx] = (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._a74[idx] = 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._a75[idx] = (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._a76[idx] = (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

    def new_step_init(self, u: np.ndarray) -> None:
        """Initialize nonlinear term for new step."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform one ETD35 stage pass for a diagonal system."""
        if accept:
            self.new_step_init(u)

        self._k = self._EL14 * u + self._a21 * self._NL1
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL14 * u + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL12 * u + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL34 * u + self._a51 * self._NL1 + self._a52 * (self._NL2 - self._NL3) + self._a54 * self._NL4
        self._NL5 = self.nl_func(self._k)
        self._k = (
            self._EL * u
            + self._a61 * self._NL1
            + self._a62 * (self._NL2 - 3 * self._NL4 / 2.0)
            + self._a63 * self._NL3
            + self._a65 * self._NL5
        )
        self._NL6 = self.nl_func(self._k)
        self._k = (
            self._EL * u
            + self._a71 * self._NL1
            + self._a73 * self._NL3
            + self._a74 * self._NL4
            + self._a75 * self._NL5
            + self._a76 * self._NL6
        )
        self._err = self._a75 * (-self._NL1 + 4 * self._NL3 - 6 * self._NL4 + 4 * self._NL5 - self._NL6)
        return self._k, self._err


class _Etd35Diagonalized(_Etd35Diagonal):
    """ETD35 solver for non-diagonal systems via eigenvector diagonalization."""

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """Initialize diagonalized ETD35 solver."""
        super().__init__(lin_op, nl_func, etd_config)
        if len(lin_op.shape) == 1:
            raise ValueError("Cannot diagonalize a 1D system")
        lin_op_cond = np.linalg.cond(lin_op)
        if lin_op_cond > 1e16:
            raise ValueError("Linear operator is non-invertible")
        if lin_op_cond > 1000:
            self.logger.warning(
                f"Linear matrix array has a large condition number of {lin_op_cond:.2f}, method may be unstable"
            )
        self._eig_vals, self._S = np.linalg.eig(lin_op)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(lin_op.shape[0])

    def update_coeffs(self, h: float) -> None:
        """Update coefficients using eigenvalues."""
        z = h * self._eig_vals
        self._update_coeffs_diagonal(h, z)

    def new_step_init(self, u: np.ndarray) -> None:
        """Initialize transformed nonlinear term for new step."""
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform one ETD35 stage pass for diagonalized systems."""
        if accept:
            self.new_step_init(u)

        self._k = self._EL14 * self._v + self._a21 * self._NL1
        self._NL2 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL14 * self._v + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL12 * self._v + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = (
            self._EL34 * self._v + self._a51 * self._NL1 + self._a52 * (self._NL2 - self._NL3) + self._a54 * self._NL4
        )
        self._NL5 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = (
            self._EL * self._v
            + self._a61 * self._NL1
            + self._a62 * (self._NL2 - 3 * self._NL4 / 2.0)
            + self._a63 * self._NL3
            + self._a65 * self._NL5
        )
        self._NL6 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = (
            self._EL * self._v
            + self._a71 * self._NL1
            + self._a73 * self._NL3
            + self._a74 * self._NL4
            + self._a75 * self._NL5
            + self._a76 * self._NL6
        )
        self._err = self._a75 * (-self._NL1 + 4 * self._NL3 - 6 * self._NL4 + 4 * self._NL5 - self._NL6)
        return self._S.dot(self._k), self._err


class _Etd35NonDiagonal:
    """ETD35 solver for full (non-diagonal, non-diagonalizable) linear operators."""

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """Initialize ETD35 for non-diagonal systems."""
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL14, self._EL12, self._EL34, self._EL = [
            np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(4)
        ]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
            self._a61,
            self._a62,
            self._a63,
            self._a64,
            self._a65,
            self._a71,
            self._a73,
            self._a74,
            self._a75,
            self._a76,
        ) = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(18)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:  # pylint: disable=too-many-locals
        """Update ETD coefficients for given step size `h`."""
        z = h * self.lin_op
        # Use expm matrix exponential function from scipy
        self._EL14 = expm(z / 4.0)
        self._EL12 = expm(z / 2.0)
        self._EL34 = expm(3.0 * z / 4.0)
        self._EL = expm(z)

        # Use contour integral evaluation for psi etd functions
        contour_points = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        phi1_14, phi2_14, phi1_12, phi2_12, phi1_34, phi2_34 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(6)
        ]
        phi1_1, phi2_1, phi3_1 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(3)]

        for point in contour_points:
            Q14 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 4.0)
            Q12 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 2.0)
            Q34 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - 3 * z / 4.0)
            Q = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z)
            phi1_14 += point * phi1(point) * Q14 / self.etd_config.contour_points
            phi2_14 += point * phi2(point) * Q14 / self.etd_config.contour_points
            phi1_12 += point * phi1(point) * Q12 / self.etd_config.contour_points
            phi2_12 += point * phi2(point) * Q12 / self.etd_config.contour_points
            phi1_34 += point * phi1(point) * Q34 / self.etd_config.contour_points
            phi2_34 += point * phi2(point) * Q34 / self.etd_config.contour_points
            phi1_1 += point * phi1(point) * Q / self.etd_config.contour_points
            phi2_1 += point * phi2(point) * Q / self.etd_config.contour_points
            phi3_1 += point * phi3(point) * Q / self.etd_config.contour_points

        self._a21 = h * phi1_14 / 4.0
        self._a31 = h * (phi1_14 - phi2_14 / 2.0) / 4.0
        self._a32 = h * phi2_14 / 8.0
        self._a41 = h * (phi1_12 - phi2_12) / 2.0
        self._a43 = h * phi2_12 / 2.0
        self._a51 = h * 3.0 * (phi1_34 - 3.0 * phi2_34 / 4.0) / 4.0
        self._a52 = -3 * h * phi1_34 / 8.0
        self._a54 = h * 9 * phi2_34 / 16.0
        self._a61 = h * (-77 * phi1_1 + 59 * phi2_1) / 42.0
        self._a62 = h * 8 * phi1_1 / 7.0
        self._a63 = h * (111 * phi1_1 - 87 * phi2_1) / 28.0
        self._a65 = h * (-47 * phi1_1 + 143 * phi2_1) / 84.0
        self._a71 = h * 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        self._a73 = h * (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._a74 = h * 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._a75 = h * (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._a76 = h * (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

    def new_step_init(self, u: np.ndarray) -> None:
        """Initialize nonlinear term for new time step."""
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """Perform one ETD35 Runge–Kutta update for non-diagonal systems."""
        if accept:
            self.new_step_init(u)

        self._k = self._EL14.dot(u) + self._a21.dot(self._NL1)
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL14.dot(u) + self._a31.dot(self._NL1) + self._a32.dot(self._NL2)
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL12.dot(u) + self._a41.dot(self._NL1) + self._a43.dot(self._NL3)
        self._NL4 = self.nl_func(self._k)
        self._k = (
            self._EL34.dot(u)
            + self._a51.dot(self._NL1)
            + self._a52.dot(self._NL2 - self._NL3)
            + self._a54.dot(self._NL4)
        )
        self._NL5 = self.nl_func(self._k)
        self._k = (
            self._EL.dot(u)
            + self._a61.dot(self._NL1)
            + self._a62.dot(self._NL2 - 3 * self._NL4 / 2.0)
            + self._a63.dot(self._NL3)
            + self._a65.dot(self._NL5)
        )
        self._NL6 = self.nl_func(self._k)
        self._k = (
            self._EL.dot(u)
            + self._a71.dot(self._NL1)
            + self._a73.dot(self._NL3)
            + self._a74.dot(self._NL4)
            + self._a75.dot(self._NL5)
            + self._a76.dot(self._NL6)
        )
        self._err = self._a75.dot(-self._NL1 + 4 * self._NL3 - 6 * self._NL4 + 4 * self._NL5 - self._NL6)
        return self._k, self._err


class ETD35(ETDAS):
    """
    Fifth-order Exponential Time Differencing solver with adaptive stepping (ETD(3,5)).

    This class implements the ETD(3,5) scheme for stiff systems of the form:
        dU/dt = L·U + N(U)

    It adapts the time step automatically using an embedded third-order method
    for local error control, via the adaptive machinery of :class:`ETDAS`.

    Attributes
    ----------
    lin_op : np.ndarray
        Linear operator `L` in the system. May be a 1D array (diagonal system)
        or a 2D square matrix (non-diagonal system).

    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function `N(U)`.

    diagonalize : bool
        If True, attempts eigenvalue decomposition to transform system into
        diagonal form before solving.

    Notes
    -----
    Solver behavior (safety factors, error tolerances, etc.) is governed by
    :class:`SolverConfig` and :class:`ETDConfig`.
    """

    _method: Union[_Etd35Diagonal, _Etd35Diagonalized, _Etd35NonDiagonal]

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        etd_config: ETDConfig = ETDConfig(),
        diagonalize: bool = False,
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ):
        """Initialize ETD35 adaptive solver."""
        super().__init__(
            lin_op,
            nl_func,
            config=config,
            etd_config=etd_config,
            loglevel=loglevel,
        )
        if self._diag:
            self._method = _Etd35Diagonal(lin_op, nl_func, self.etd_config)
        else:
            if diagonalize:
                self._method = _Etd35Diagonalized(lin_op, nl_func, self.etd_config)
            else:
                self._method = _Etd35NonDiagonal(lin_op, nl_func, self.etd_config)
        self._stages_init = False
        self._accept = False

    def _reset(self) -> None:
        """Reset internal solver state for restart or initialization."""
        self._stages_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """Update ETD35 coefficients if time step has changed."""
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD35 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute next-step state using one ETD35 Runge–Kutta pass.

        Parameters
        ----------
        u : np.ndarray
            Current state vector `u_n`.
        h : float
            Time step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The tuple `(u_next, err_estimate)` with the next-step value and
            the local truncation error estimate.
        """
        self._update_coeffs(h)
        if not self._stages_init:
            self._method.new_step_init(u)
            self._stages_init = True
        return self._method.update_stages(u, self._accept)

    def _q(self) -> int:
        """Order variable for computing suggested step size (embedded order + 1)"""
        return 4
