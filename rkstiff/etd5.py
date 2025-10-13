"""rkstiff.etd5: Exponential time-differencing constant step solver of 5th order."""

from typing import Callable, Union
import numpy as np
from scipy.linalg import expm
from rkstiff.etd import ETDCS, ETDConfig, phi1, phi2, phi3


class _Etd5Diagonal: # pylint: disable=too-few-public-methods
    """
    ETD5 diagonal system strategy for ETD5 solver.

    Optimized implementation for diagonal linear operators that avoids
    matrix operations by using element-wise operations on vectors.

    Parameters
    ----------
    lin_op : np.ndarray
        1D array representing the diagonal linear operator.
    nl_func : callable
        Nonlinear function nl_func(U) that maps np.ndarray -> np.ndarray.
    etd_config : ETDConfig
        Configuration object containing modecutoff, contour_points,
        and contour_radius parameters.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
    ) -> None:
        """Initialize ETD5 diagonal system strategy."""
        self.lin_op = lin_op
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
        ) = [np.zeros(n, dtype=np.complex128) for _ in range(13)]
        self._b1, self._b3, self._b4, self._b5, self._b6 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """
        Update ETD5 coefficients based on step size h.

        Computes exponential and phi function coefficients for the diagonal system.
        Small modes (|h*λ| < modecutoff) use contour integration to avoid numerical
        instability, while large modes use direct evaluation.

        Parameters
        ----------
        h : float
            Time step size.
        """

        z = h * self.lin_op
        # diagonal system -> L is 1D array (of independent modes)
        self._EL14 = np.exp(z / 4.0)
        self._EL12 = np.exp(z / 2.0)
        self._EL34 = np.exp(3.0 * z / 4.0)
        self._EL = np.exp(z)

        smallmode_idx = np.abs(z) < self.etd_config.modecutoff
        # compute big mode coeffs
        zb = z[~smallmode_idx]  # z big
        phi1_14 = h * phi1(zb / 4)
        phi2_14 = h * phi2(zb / 4)
        phi1_12 = h * phi1(zb / 2)
        phi2_12 = h * phi2(zb / 2)
        phi1_34 = h * phi1(3 * zb / 4)
        phi2_34 = h * phi2(3 * zb / 4)
        phi1_1 = h * phi1(zb)
        phi2_1 = h * phi2(zb)
        phi3_1 = h * phi3(zb)

        self._a21[~smallmode_idx] = phi1_14 / 4.0
        self._a31[~smallmode_idx] = (phi1_14 - phi2_14 / 2.0) / 4.0
        self._a32[~smallmode_idx] = phi2_14 / 8.0
        self._a41[~smallmode_idx] = (phi1_12 - phi2_12) / 2.0
        self._a43[~smallmode_idx] = phi2_12 / 2.0
        self._a51[~smallmode_idx] = 3.0 * (phi1_34 - 3.0 * phi2_34 / 4.0) / 4.0
        self._a52[~smallmode_idx] = -3 * phi1_34 / 8.0
        self._a54[~smallmode_idx] = 9 * phi2_34 / 16.0
        self._a61[~smallmode_idx] = (-77 * phi1_1 + 59 * phi2_1) / 42.0
        self._a62[~smallmode_idx] = 8 * phi1_1 / 7.0
        self._a63[~smallmode_idx] = (111 * phi1_1 - 87 * phi2_1) / 28.0
        self._a65[~smallmode_idx] = (-47 * phi1_1 + 143 * phi2_1) / 84.0
        self._b1[~smallmode_idx] = 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        # Paper has error in b3 phi2 coefficient (states this is 497 but it is actually 467)
        self._b3[~smallmode_idx] = (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._b4[~smallmode_idx] = 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._b5[~smallmode_idx] = (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._b6[~smallmode_idx] = (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

        # compute small mode coeffs
        zs = z[smallmode_idx]  # z small
        r = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        phi1_14 = h * np.sum(phi1(Z / 4), axis=1) / self.etd_config.contour_points
        phi2_14 = h * np.sum(phi2(Z / 4), axis=1) / self.etd_config.contour_points
        phi1_12 = h * np.sum(phi1(Z / 2), axis=1) / self.etd_config.contour_points
        phi2_12 = h * np.sum(phi2(Z / 2), axis=1) / self.etd_config.contour_points
        phi1_34 = h * np.sum(phi1(3 * Z / 4), axis=1) / self.etd_config.contour_points
        phi2_34 = h * np.sum(phi2(3 * Z / 4), axis=1) / self.etd_config.contour_points
        phi1_1 = h * np.sum(phi1(Z), axis=1) / self.etd_config.contour_points
        phi2_1 = h * np.sum(phi2(Z), axis=1) / self.etd_config.contour_points
        phi3_1 = h * np.sum(phi3(Z), axis=1) / self.etd_config.contour_points

        self._a21[smallmode_idx] = phi1_14 / 4.0
        self._a31[smallmode_idx] = (phi1_14 - phi2_14 / 2.0) / 4.0
        self._a32[smallmode_idx] = phi2_14 / 8.0
        self._a41[smallmode_idx] = (phi1_12 - phi2_12) / 2.0
        self._a43[smallmode_idx] = phi2_12 / 2.0
        self._a51[smallmode_idx] = 3.0 * (phi1_34 - 3.0 * phi2_34 / 4.0) / 4.0
        self._a52[smallmode_idx] = -3 * phi1_34 / 8.0
        self._a54[smallmode_idx] = 9 * phi2_34 / 16.0
        self._a61[smallmode_idx] = (-77 * phi1_1 + 59 * phi2_1) / 42.0
        self._a62[smallmode_idx] = 8 * phi1_1 / 7.0
        self._a63[smallmode_idx] = (111 * phi1_1 - 87 * phi2_1) / 28.0
        self._a65[smallmode_idx] = (-47 * phi1_1 + 143 * phi2_1) / 84.0
        self._b1[smallmode_idx] = 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        # Paper has error in b3 phi2 coefficient (states this is 497 but it is actually 467)
        self._b3[smallmode_idx] = (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._b4[smallmode_idx] = 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._b5[smallmode_idx] = (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._b6[smallmode_idx] = (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize the first nonlinear evaluation.

        Stores nl_func(u_n) for use in the first stage. This implements the
        "First Same As Last" (FSAL) principle for efficiency.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray) -> np.ndarray:
        """
        Advance solution by one time step using six-stage ETD5 scheme.

        Executes the six Runge-Kutta-like stages of the ETD5 method with
        element-wise operations optimized for diagonal systems.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector u_n.

        Returns
        -------
        np.ndarray
            Updated solution vector u_{n+1}.
        """
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
            + self._b1 * self._NL1
            + self._b3 * self._NL3
            + self._b4 * self._NL4
            + self._b5 * self._NL5
            + self._b6 * self._NL6
        )
        self._NL1 = self._NL6  # FSAL principle
        return self._k


class _Etd5NonDiagonal: # pylint: disable=too-few-public-methods
    """
    ETD5 non-diagonal system strategy for ETD5 solver.

    General implementation for full matrix linear operators using
    matrix-vector products and contour integration for all modes.

    Parameters
    ----------
    lin_op : np.ndarray
        2D array representing the full matrix linear operator.
    nl_func : callable
        Nonlinear function nl_func(U) that maps np.ndarray -> np.ndarray.
    etd_config : ETDConfig
        Configuration object containing contour_points and contour_radius
        parameters (modecutoff is not used for non-diagonal systems).
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """Initialize ETD5 non-diagonal system strategy."""
        self.lin_op = lin_op
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
        ) = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(13)]
        self._b1, self._b3, self._b4, self._b5, self._b6 = [
            np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)

    def _update_coeffs(self, h: float) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """
        Update ETD5 coefficients for non-diagonal systems.

        Computes exponential and phi function coefficients using contour integration
        for all modes. Uses matrix exponentials and Cauchy integral formula.

        Parameters
        ----------
        h : float
            Time step size.
        """
        # Update RK coefficients for 'matrix' np.array operator lin_op
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
        self._b1 = h * 7 * (257 * phi1_1 - 497 * phi2_1 + 270 * phi3_1) / 2700
        self._b3 = h * (1097 * phi1_1 - 467 * phi2_1 - 150 * phi3_1) / 1350
        self._b4 = h * 2 * (-49 * phi1_1 + 199 * phi2_1 - 135 * phi3_1) / 225
        self._b5 = h * (-313 * phi1_1 + 883 * phi2_1 - 90 * phi3_1) / 1350
        self._b6 = h * (509 * phi1_1 - 2129 * phi2_1 + 1830 * phi3_1) / 2700

    def _n1_init(self, u: np.ndarray) -> None:
        """
        Initialize the first nonlinear evaluation.

        Stores nl_func(u_n) for use in the first stage. This implements the
        "First Same As Last" (FSAL) principle for efficiency.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def _update_stages(self, u: np.ndarray) -> np.ndarray:
        """
        Advance solution by one time step using six-stage ETD5 scheme.

        Executes the six Runge-Kutta-like stages of the ETD5 method using
        matrix-vector products for non-diagonal systems.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector u_n.

        Returns
        -------
        np.ndarray
            Updated solution vector u_{n+1}.
        """
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
            + self._b1.dot(self._NL1)
            + self._b3.dot(self._NL3)
            + self._b4.dot(self._NL4)
            + self._b5.dot(self._NL5)
            + self._b6.dot(self._NL6)
        )
        self._NL1 = self._NL6  # FSAL principle
        return self._k


class ETD5(ETDCS):
    """
    Fifth-order exponential time-differencing solver with constant step size.

    Implements a 5th-order ETD scheme for semi-linear systems dU/dt = L*U + NL(U),
    where L is a linear operator and nl_func is a nonlinear function.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator L. Can be 1D (diagonal) or 2D (full matrix).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function nl_func(U).
    etd_config : ETDConfig, optional
        Configuration for ETD parameters (modecutoff, contour_points, contour_radius).

    Notes
    -----
    Automatically selects optimized implementation based on operator structure.
    Uses contour integration for computing matrix exponentials and phi functions.
    """
    _method: Union[_Etd5Diagonal, _Etd5NonDiagonal]

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
    ) -> None:
        """
        Initialize the ETD5 solver.

        Sets up the fifth-order exponential time-differencing solver by configuring
        the linear and nonlinear components, and selecting the appropriate internal
        method (diagonal or non-diagonal) based on the structure of the linear operator.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator (L) in the system dU/dt = L*U + NL(U).
            - If 1D array: treated as diagonal operator (more efficient)
            - If 2D array: treated as full matrix operator
            Supports both real and complex values.
        nl_func : callable
            Nonlinear function nl_func(U) that maps np.ndarray -> np.ndarray.
            Takes the current state vector and returns the nonlinear contribution.
            Must have signature: nl_func(u: np.ndarray) -> np.ndarray
        etd_config : ETDConfig, default=ETDConfig()
            Configuration object containing:
            - modecutoff (float): Threshold for small eigenvalue modes.
              Eigenvalues with |h*λ| < modecutoff use Taylor series expansions
              instead of direct evaluation to avoid numerical instability in
              phi functions (diagonal systems only).
            - contour_points (int): Number of quadrature points for contour
              integration when computing matrix exponentials and phi functions.
              More points increase accuracy but also computational cost.
            - contour_radius (float): Radius of the circular contour in the
              complex plane used for computing matrix functions via Cauchy
              integral formula. Should be chosen to properly enclose the
              spectrum of h*L where h is the time step.

        Notes
        -----
        - The solver automatically detects if lin_op is 1D (diagonal) and selects
          the optimized _Etd5Diagonal method; otherwise uses _Etd5NonDiagonal.
        - Internal state variables are initialized but coefficients are not computed
          until the first time step is taken.
        - The parent class ETDCS handles common setup and validation.
        """
        super().__init__(lin_op, nl_func, etd_config)
        if self._diag:
            self._method = _Etd5Diagonal(lin_op, nl_func, self.etd_config)
        else:
            self._method = _Etd5NonDiagonal(lin_op, nl_func, self.etd_config)
        self.__n1_init = False

    def _reset(self) -> None:
        """
        Reset the solver to its initial state.

        Clears all cached coefficients and internal state variables, returning
        the solver to the state immediately after initialization. This is useful
        when switching to a different initial condition or when restarting a
        simulation.

        Notes
        -----
        - Clears the initialization flag, forcing reinitialization on next step
        - Removes cached step size coefficients
        - Does not affect the linear operator, nonlinear function, or
          configuration settings
        """
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD5 coefficients if the step size has changed.

        Computes and caches the exponential and phi function coefficients required
        for the ETD5 method. Coefficients depend on h*L and are expensive to compute,
        so they are only recalculated when the step size changes.

        Parameters
        ----------
        h : float
            Time step size. Must be positive.

        Notes
        -----
        - If h equals the cached step size (self._h_coeff), returns immediately
          without recomputing
        - Delegates actual computation to the internal method object
          (_Etd5Diagonal or _Etd5NonDiagonal)
        - Logs coefficient updates for debugging and monitoring
        - The coefficients include matrix exponentials exp(h*L/4), exp(h*L/2),
          exp(3h*L/4), exp(h*L) and phi functions φ₁, φ₂, φ₃ evaluated at
          various fractional steps
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logs.append("ETD5 coefficients updated")

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance the solution by one time step using the ETD5 scheme.

        Computes u_{n+1} from u_n by executing the six-stage Runge-Kutta-like
        procedure of the ETD5 method. This includes evaluating the nonlinear
        function at intermediate stages and combining results using precomputed
        exponential and phi function coefficients.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector u_n at time t_n.
        h : float
            Time step size Δt = t_{n+1} - t_n. Must be positive.

        Returns
        -------
        np.ndarray
            Updated solution vector u_{n+1} at time t_{n+1} = t_n + h.

        Notes
        -----
        - Automatically updates coefficients if h has changed since last call
        - On the first call, initializes internal state by evaluating nl_func(u) and
          storing it for subsequent steps (FSAL principle)
        - The method uses six nonlinear function evaluations per step for
          5th-order accuracy
        - Delegates the actual stage computations to the internal method object
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u)
