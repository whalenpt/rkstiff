r"""
Constant-Step Fourth-Order Exponential Time-Differencing Integrator
===================================================================

Implements a **fourth-order exponential time-differencing (ETD4) solver**
for stiff partial differential equations (PDEs) of the form

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
      = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is a linear spatial differential operator
(e.g. Laplacian, biharmonic, etc.), and :math:`\mathcal{N}(\mathbf{U})`
is a nonlinear term in physical or spectral space.

The solver advances the field :math:`\mathbf{U}(x, t)` in time using
exponential Runge–Kutta (Krogstad, 2005) methods.

References
----------
Krogstad, S. (2005).
*Generalized integrating factor methods for stiff PDEs.*
Journal of Computational Physics, 203(1), 72–88.
"""

import logging
from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .etd import ETDCS, ETDConfig, psi1, psi2, psi3


class _Etd4Diagonal:  # pylint: disable=too-few-public-methods
    r"""
    ETD4 solver for diagonalized PDE systems.

    Solves evolution equations of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` acts diagonally in Fourier or spectral space.
    The time step updates are performed using exponential integrators:

    .. math::

        \mathbf{U}(t + h)
          = e^{h\mathcal{L}}\mathbf{U}(t)
            + \sum_{r=1}^3 h\,b_r(h\mathcal{L})\,\mathcal{N}_r(\mathbf{U}).

    The :math:`\psi_r(z)` functions used in the coefficients are defined as

    .. math::

        \psi_r(z)
          = r \int_0^1 e^{(1-\theta)z}\,\theta^{r-1}\,d\theta,
          \quad r = 1,2,3,\dots
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
        psi1_12 = h * psi1(zb / 2)
        psi2_12 = h * psi2(zb / 2)
        psi1_1 = h * psi1(zb)
        psi2_1 = h * psi2(zb)
        psi3_1 = h * psi3(zb)

        self._a21[~smallmode_idx] = 0.5 * psi1_12
        self._a31[~smallmode_idx] = 0.5 * (psi1_12 - psi2_12)
        self._a32[~smallmode_idx] = 0.5 * psi2_12
        self._a41[~smallmode_idx] = psi1_1 - psi2_1
        self._a43[~smallmode_idx] = psi2_1
        self._b1[~smallmode_idx] = psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1
        self._b2[~smallmode_idx] = psi2_1 - (2.0 / 3) * psi3_1
        self._b4[~smallmode_idx] = -(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1

        # compute small mode coeffs
        zs = z[smallmode_idx]  # z small
        r = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        psi1_12 = h * np.sum(psi1(Z / 2), axis=1) / self.etd_config.contour_points
        psi2_12 = h * np.sum(psi2(Z / 2), axis=1) / self.etd_config.contour_points
        psi1_1 = h * np.sum(psi1(Z), axis=1) / self.etd_config.contour_points
        psi2_1 = h * np.sum(psi2(Z), axis=1) / self.etd_config.contour_points
        psi3_1 = h * np.sum(psi3(Z), axis=1) / self.etd_config.contour_points

        self._a21[smallmode_idx] = 0.5 * psi1_12
        self._a31[smallmode_idx] = 0.5 * (psi1_12 - psi2_12)
        self._a32[smallmode_idx] = 0.5 * psi2_12
        self._a41[smallmode_idx] = psi1_1 - psi2_1
        self._a43[smallmode_idx] = psi2_1
        self._b1[smallmode_idx] = psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1
        self._b2[smallmode_idx] = psi2_1 - (2.0 / 3) * psi3_1
        self._b4[smallmode_idx] = -(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1

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
    r"""
    ETD4 solver for non-diagonal PDE operators.

    Suitable for PDEs where the linear operator :math:`\mathcal{L}`
    couples multiple spatial modes, such as systems with mixed derivatives.

    The governing equation is

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    The update is performed using matrix exponentials and the auxiliary
    :math:`\psi_r(h\mathcal{L})` functions:

    .. math::

        \psi_r(h\mathcal{L})
          = r \int_0^1 e^{(1-\theta)h\mathcal{L}} \theta^{r-1}\,d\theta.

    For small spectral radii, the integral is approximated by complex contour
    quadrature for numerical stability.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
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

        psi1_12, psi2_12, psi1_1, psi2_1, psi3_1 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        for point in contour_points:
            Q = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z)
            Q2 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 2)
            psi1_12 += point * psi1(point) * Q2 / self.etd_config.contour_points
            psi2_12 += point * psi2(point) * Q2 / self.etd_config.contour_points
            psi1_1 += point * psi1(point) * Q / self.etd_config.contour_points
            psi2_1 += point * psi2(point) * Q / self.etd_config.contour_points
            psi3_1 += point * psi3(point) * Q / self.etd_config.contour_points

        self._a21 = 0.5 * h * psi1_12
        self._a31 = 0.5 * h * (psi1_12 - psi2_12)
        self._a32 = 0.5 * h * psi2_12
        self._a41 = h * (psi1_1 - psi2_1)
        self._a43 = h * psi2_1
        self._b1 = h * (psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1)
        self._b2 = h * (psi2_1 - (2.0 / 3) * psi3_1)
        self._b4 = h * (-(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1)

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
    Fourth-order Exponential Time-Differencing solver for PDEs.

    Integrates stiff PDEs of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    where
    - :math:`\mathcal{L}` is the linear spatial operator (e.g. diffusion),
    - :math:`\mathcal{N}` represents nonlinear stiff terms.

    The solution is advanced in time using a four-stage exponential
    Runge-Kutta (Krogstad) scheme, with precomputed
    :math:`\psi_r` coefficients.

    Parameters
    ----------
    lin_op : np.ndarray
        Discretized linear operator :math:`\mathcal{L}` in matrix or diagonal form.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear functional :math:`\mathcal{N}[\mathbf{U}]`.
    etd_config : ETDConfig, optional
        Configuration for contour integration and cutoff parameters.

    Notes
    -----
    This solver is designed for PDEs discretized in space, e.g.
    after applying a Fourier spectral or finite-difference transform.
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
