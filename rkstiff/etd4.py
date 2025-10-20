"""rkstiff.etd4: Exponential time-differencing constant step solver of 4th order (Krogstad)"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.etd import ETDCS, ETDConfig, phi1, phi2, phi3


class _Etd4Diagonal:  # pylint: disable=too-few-public-methods
    """
    ETD4 diagonal system strategy for ETD4 solver.

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

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig) -> None:
        """Initialize ETD4 diagonal system strategy"""
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        self._a21, self._a31, self._a32, self._a41, self._a43 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._b1, self._b2, self._b4 = [np.zeros(n, dtype=np.complex128) for _ in range(3)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """
        Update ETD4 coefficients based on step size h.

        Computes exponential and phi function coefficients for the diagonal system.
        Small modes (|h*λ| < modecutoff) use contour integration to avoid numerical
        instability, while large modes use direct evaluation.

        Parameters
        ----------
        h : float
            Time step size.
        """
        z = h * self.lin_op
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
        self._b1[~smallmode_idx] = phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1
        self._b2[~smallmode_idx] = phi2_1 - (2.0 / 3) * phi3_1
        self._b4[~smallmode_idx] = -(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1

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
        self._b1[smallmode_idx] = phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1
        self._b2[smallmode_idx] = phi2_1 - (2.0 / 3) * phi3_1
        self._b4[smallmode_idx] = -(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1

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
        Advance solution by one time step using four-stage ETD4 scheme.

        Executes the four Runge-Kutta-like stages of the ETD4 method with
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
        # Use First is same as last principle (FSAL)
        self._k = self._EL2 * u + self._a21 * self._NL1
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2 * u + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL * u + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL * u + self._b1 * self._NL1 + self._b2 * (self._NL2 + self._NL3) + self._b4 * self._NL4
        self._NL1 = self.nl_func(self._k)
        return self._k


class _Etd4NonDiagonal:  # pylint: disable=too-few-public-methods
    """
    ETD4 non-diagonal system strategy for ETD4 solver.

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

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig) -> None:
        """
        Initialize ETD4 non-diagonal system strategy.

        Parameters
        ----------
        lin_op : np.ndarray
            2D array representing the full matrix linear operator.
        nl_func : callable
            Nonlinear function with signature: nl_func(u: np.ndarray) -> np.ndarray
        etd_config : ETDConfig
            Configuration parameters for the ETD scheme.
        """
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(2)]
        self._a21, self._a31, self._a32, self._a41, self._a43 = [
            np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        self._b1, self._b2, self._b4 = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(3)]
        self._NL1, self._NL2, self._NL3, self._NL4 = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """
        Update ETD4 coefficients based on step size h.

        Computes matrix exponentials and phi function matrices using contour
        integration for the full non-diagonal system.

        Parameters
        ----------
        h : float
            Time step size.
        """
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
        self._b1 = h * (phi1_1 - (3.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1)
        self._b2 = h * (phi2_1 - (2.0 / 3) * phi3_1)
        self._b4 = h * (-(1.0 / 2) * phi2_1 + (2.0 / 3) * phi3_1)

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
        Advance solution by one time step using four-stage ETD4 scheme.

        Executes the four Runge-Kutta-like stages of the ETD4 method using
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
        self._k = self._EL2.dot(u) + self._a21.dot(self._NL1)
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2.dot(u) + self._a31.dot(self._NL1) + self._a32.dot(self._NL2)
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL.dot(u) + self._a41.dot(self._NL1) + self._a43.dot(self._NL3)
        self._NL4 = self.nl_func(self._k)
        self._k = (
            self._EL.dot(u) + self._b1.dot(self._NL1) + self._b2.dot(self._NL2 + self._NL3) + self._b4.dot(self._NL4)
        )
        self._NL1 = self.nl_func(self._k)  # Use First is same as last principle (FSAL)
        return self._k


class ETD4(ETDCS):
    """
    Fourth-order exponential time-differencing solver with constant step size.

    This class implements Krogstad's 4th-order ETD scheme for solving semi-linear
    differential equations of the form dU/dt = L*U + NL(U), where L is a linear
    operator and NL is a nonlinear function. The method uses contour integration
    for computing matrix exponentials and phi functions.

    The solver maintains internal state and automatically updates coefficients when
    the step size changes. It supports both diagonal and non-diagonal linear operators,
    automatically selecting the appropriate optimized implementation.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator (L) in the system dU/dt = L*U + NL(U). Can be:
        - 2D array: full matrix representation
        - 1D array: diagonal system (more efficient)
        Both real-valued and complex-valued operators are supported.
    nl_func : callable
        Nonlinear function NL(U) that maps np.ndarray -> np.ndarray.
        Accepts the current state and returns the nonlinear contribution.
        Can be real-valued or complex-valued.
    etd_config : ETDConfig, default=ETDConfig()
        Configuration object containing ETD scheme parameters:
        - modecutoff (float): Threshold for small eigenvalue modes (default=0.01)
        - contour_points (int): Number of contour integration points (default=32)
        - contour_radius (float): Contour radius in complex plane (default=1.0)

    Attributes
    ----------
    lin_op : np.ndarray
        Stored linear operator.
    nl_func : callable
        Stored nonlinear function.
    etd_config : ETDConfig
        Configuration parameters for the ETD scheme.
    t : np.ndarray
        Time values from the most recent call to evolve() method.
    u : np.ndarray
        Solution array from the most recent call to evolve() method.

    Notes
    -----
    - The solver automatically detects whether the linear operator is diagonal and
      uses optimized routines accordingly (_Etd4Diagonal vs _Etd4NonDiagonal).
    - Coefficients are cached and only recomputed when the step size changes.
    - The first step initializes internal state for the multi-stage Runge-Kutta scheme.
    - For diagonal systems, modes with |h*λ| < modecutoff use contour integration
      to avoid numerical instability in phi function evaluation.

    References
    ----------
    Krogstad, S. (2005). Generalized integrating factor methods for stiff PDEs.
    Journal of Computational Physics, 203(1), 72-88.

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.etd import ETDConfig
    >>> # Define a simple system: dU/dt = -U + U^2
    >>> linear_op = np.array([[-1.0]])
    >>> nl_func = lambda u: u**2
    >>> config = ETDConfig(modecutoff=0.01, contour_points=32, contour_radius=1.0)
    >>> solver = ETD4(linear_op, nl_func, config)
    >>> # Solve from t=0 to t=1 with step size 0.01
    >>> u0 = np.array([0.5])
    >>> solver.evolve(u0, 0.0, 1.0, 0.01)

    See Also
    --------
    ETDCS : Parent class for constant-step ETD methods
    ETDConfig : Configuration dataclass for ETD parameters
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize the ETD4 solver.

        Sets up the fourth-order exponential time-differencing solver by configuring
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
            Nonlinear function NL(U) that maps np.ndarray -> np.ndarray.
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
          the optimized _Etd4Diagonal method; otherwise uses _Etd4NonDiagonal.
        - Internal state variables are initialized but coefficients are not computed
          until the first time step is taken.
        - The parent class ETDCS handles common setup and validation.
        """
        super().__init__(lin_op, nl_func, etd_config=etd_config, loglevel=loglevel)
        self._method = Union[_Etd4Diagonal, _Etd4NonDiagonal]
        if self._diag:
            self._method = _Etd4Diagonal(lin_op, nl_func, self.etd_config)
        else:
            self._method = _Etd4NonDiagonal(lin_op, nl_func, self.etd_config)
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
        Update ETD4 coefficients if the step size has changed.

        Computes and caches the exponential and phi function coefficients required
        for the ETD4 method. Coefficients depend on h*L and are expensive to compute,
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
          (_Etd4Diagonal or _Etd4NonDiagonal)
        - Logs coefficient updates for debugging and monitoring
        - The coefficients include matrix exponentials exp(h*L), exp(h*L/2) and
          phi functions φ₁, φ₂, φ₃ evaluated at h*L and h*L/2
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD4 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance the solution by one time step using the ETD4 scheme.

        Computes u_{n+1} from u_n by executing the four-stage Runge-Kutta-like
        procedure of the ETD4 method. This includes evaluating the nonlinear
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
        - The method uses four nonlinear function evaluations per step for
          4th-order accuracy
        - Delegates the actual stage computations to the internal method object
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u)
