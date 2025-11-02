"""rkstiff.etd4: Exponential time-differencing constant step solver of 4th order (Krogstad)"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from rkstiff.etd import ETDCS, ETDConfig, phi1, phi2, phi3


class _Etd4Diagonal:  # pylint: disable=too-few-public-methods
    """
    ETD4 diagonal system strategy for ETD4 solver.

    Parameters
    ----------
    lin_op : np.ndarray
        1D array representing the diagonal linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
    etd_config : ETDConfig
        Configuration object containing modecutoff, contour_points, and contour_radius parameters.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig) -> None:
        """
        Initialize ETD4 diagonal system strategy.

        Parameters
        ----------
        lin_op : np.ndarray
            Diagonal linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration.
        """
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

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray) -> np.ndarray:
        """
        Advance solution by one time step using four-stage ETD4 scheme.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.

        Returns
        -------
        np.ndarray
            Updated solution vector.
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

    Parameters
    ----------
    lin_op : np.ndarray
        2D array representing the full matrix linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
    etd_config : ETDConfig
        Configuration object containing contour_points and contour_radius parameters.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig) -> None:
        """
        Initialize ETD4 non-diagonal system strategy.

        Parameters
        ----------
        lin_op : np.ndarray
            Full matrix linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration.
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

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray) -> np.ndarray:
        """
        Advance solution by one time step using four-stage ETD4 scheme.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.

        Returns
        -------
        np.ndarray
            Updated solution vector.
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
    r"""
    Fourth-order exponential time-differencing solver with constant step size.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator in the system dU/dt = L·U + N(U).
        Can be either a 2D matrix (full operator) or 1D array (diagonal).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function in the system dU/dt = L·U + N(U).
    etd_config : ETDConfig, optional
        ETD configuration. Default is ETDConfig().
    loglevel : str or int, optional
        Logging verbosity level. Default is "WARNING".

    Notes
    -----
    - Automatically detects whether the linear operator is diagonal and uses optimized routines accordingly.
    - Coefficients are cached and only recomputed when the step size changes.
    - The first step initializes internal state for the multi-stage Runge–Kutta scheme.
    - For diagonal systems, modes with small :math:`|h \lambda| < \text{modecutoff}`
      use contour integration to avoid numerical instability in phi function evaluation.

    References
    ----------
    Krogstad, S. (2005). *Generalized integrating factor methods for stiff PDEs.*
    Journal of Computational Physics, 203(1), 72–88.

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.etd import ETDConfig
    >>> linear_op = np.array([[-1.0]])
    >>> nl_func = lambda u: u**2
    >>> config = ETDConfig(modecutoff=0.01, contour_points=32, contour_radius=1.0)
    >>> solver = ETD4(linear_op, nl_func, config)
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
        r"""
        Initialize the ETD4 solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator in the system dU/dt = L·U + N(U).
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig, optional
            ETD configuration.
        loglevel : str or int, optional
            Logging verbosity level.
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

        Notes
        -----
        - Clears the initialization flag, forcing reinitialization on next step
        - Removes cached step size coefficients
        - Does not affect the linear operator, nonlinear function, or configuration settings
        """
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD4 coefficients if the step size has changed.

        Parameters
        ----------
        h : float
            Time step size. Must be positive.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD4 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance the solution by one time step using the ETD4 scheme.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Time step size.

        Returns
        -------
        np.ndarray
            Updated solution vector.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u)
