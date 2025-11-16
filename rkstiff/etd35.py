r"""
Adaptive-Step Fifth Order (Third Order Embedding) Exponential Time-Differencing Integrator
==========================================================================================

**Exponential Time-Differencing Runge-Kutta Integrator (ETD(3,5))**

Implements the **ETD(3,5)** exponential time-differencing scheme with
embedded third-order error estimation and adaptive step control
as described in:

    Whalen, P., Brio, M., & Moloney, J. V. (2015).
    *Exponential time-differencing with embedded Runge-Kutta adaptive step control.*
    *Journal of Computational Physics*, 280, 579-601.
    doi:[10.1016/j.jcp.2014.09.024](https://doi.org/10.1016/j.jcp.2014.09.024)

---

**Mathematical Formulation**

This solver integrates stiff systems of ODEs or semidiscretized PDEs
of the form

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is the linear (possibly stiff) operator and
:math:`\mathcal{N}` is a nonlinear term evaluated explicitly.

The ETD(3,5) scheme uses exponential Runge–Kutta stages built from
the *ψ-functions*:

.. math::

    \psi_r(z)
      = r \int_0^1 e^{(1-\theta)z}\,\theta^{r-1}\,d\theta,
      \quad r = 1,2,3,\dots

which serve as scaled exponential integrators analogous to the
:math:`\phi_r` functions in classical ETD formulations
(:math:`\psi_r = r!\,\phi_r`).

---

**Implementation Overview**

The module provides specialized strategies for different forms of
the linear operator :math:`\mathcal{L}`:

- :class:`_Etd35Diagonal` — diagonal systems (elementwise exponentials)
- :class:`_Etd35Diagonalized` — eigen-decomposed systems
- :class:`_Etd35NonDiagonal` — full matrix exponential evaluation
- :class:`ETD35` — high-level adaptive solver interface

Adaptive step control follows the embedded-order algorithm of
Whalen et al. (2015), balancing efficiency and accuracy through
local error estimates derived from the third-order embedding.

"""

import logging
from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .etd import ETDAS, ETDConfig, psi1, psi2, psi3, SolverConfig


class _Etd35Diagonal:
    r"""
    ETD(3, 5) diagonal formulation.

    Integrates

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    for diagonal :math:`\mathcal{L}` using scalar exponential updates.
    All matrix exponentials reduce to elementwise factors
    :math:`e^{h\lambda_i}`.

    Notes
    -----
    The stage coefficients are built from *psi-functions*:

    .. math::

        \psi_k(z)
        = \frac{1}{(k-1)!}\!\int_0^1 e^{(1-\theta)z}\,\theta^{k-1}\,d\theta,
        \qquad k = 1,2,3,

    which relate to the *phi-functions* often used in exponential integrators
    via :math:`\phi_k = k!\,\psi_k`.

    For small arguments :math:`|z|<\varepsilon`, contour integration is used,
    and for larger modes exact exponentials are applied.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initialize ETD35 diagonal solver state.

        Parameters
        ----------
        lin_op : np.ndarray
            Diagonal linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        logger : logging.Logger, optional
            Logger instance for this solver.
        """
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config
        self.logger = logger

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
        """
        Update ETD coefficients for given step size h.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self.lin_op
        self._update_coeffs_diagonal(h, z)

    def _update_coeffs_diagonal(self, h: float, z: np.ndarray) -> None:
        """
        Internal coefficient update for diagonal systems.

        Parameters
        ----------
        h : float
            Step size.
        z : np.ndarray
            Elementwise scaled linear operator (z = h * L).
        """
        self._EL14 = np.exp(z / 4.0)
        self._EL12 = np.exp(z / 2.0)
        self._EL34 = np.exp(3.0 * z / 4.0)
        self._EL = np.exp(z)

        smallmode_idx = np.abs(z) < self.etd_config.modecutoff

        self._update_large_mode_coeffs(h, z, smallmode_idx)
        self._update_small_mode_coeffs(h, z, smallmode_idx)

    def _update_large_mode_coeffs(self, h: float, z: np.ndarray, smallmode_idx: np.ndarray) -> None:
        """
        Compute ETD35 coefficients for modes with |z| >= cutoff.

        Parameters
        ----------
        h : float
            Step size.
        z : np.ndarray
            Elementwise scaled linear operator.
        smallmode_idx : np.ndarray
            Boolean mask for small modes.
        """
        idx = ~smallmode_idx

        # Early return if no large modes
        if not np.any(idx):
            return

        zb = z[idx]  # z big - extract large modes

        # Compute psi functions for large modes
        psi1_14, psi2_14 = h * psi1(zb / 4), h * psi2(zb / 4)
        psi1_12, psi2_12 = h * psi1(zb / 2), h * psi2(zb / 2)
        psi1_34, psi2_34 = h * psi1(3 * zb / 4), h * psi2(3 * zb / 4)
        psi1_1, psi2_1, psi3_1 = h * psi1(zb), h * psi2(zb), h * psi3(zb)

        # Assign directly to the indexed locations
        # psi1_14 etc. are already filtered, so we assign to self._a21[idx]

        self._a21[idx] = psi1_14 / 4.0
        self._a31[idx] = (psi1_14 - psi2_14 / 2.0) / 4.0
        self._a32[idx] = psi2_14 / 8.0
        self._a41[idx] = (psi1_12 - psi2_12) / 2.0
        self._a43[idx] = psi2_12 / 2.0
        self._a51[idx] = 3.0 * (psi1_34 - 3.0 * psi2_34 / 4.0) / 4.0
        self._a52[idx] = -3 * psi1_34 / 8.0
        self._a54[idx] = 9 * psi2_34 / 16.0
        self._a61[idx] = (-77 * psi1_1 + 59 * psi2_1) / 42.0
        self._a62[idx] = 8 * psi1_1 / 7.0
        self._a63[idx] = (111 * psi1_1 - 87 * psi2_1) / 28.0
        self._a65[idx] = (-47 * psi1_1 + 143 * psi2_1) / 84.0
        self._a71[idx] = 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        # Paper has error in a73/b3 psi2 coefficient (states this is 497 but it is actually 467)
        self._a73[idx] = (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._a74[idx] = 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._a75[idx] = (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._a76[idx] = (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

    def _update_small_mode_coeffs(  # pylint: disable=too-many-locals
        self, h: float, z: np.ndarray, smallmode_idx: np.ndarray
    ) -> None:
        """
        Compute ETD35 coefficients for modes with |z| < cutoff using contour integration.

        Parameters
        ----------
        h : float
            Step size.
        z : np.ndarray
            Elementwise scaled linear operator.
        smallmode_idx : np.ndarray
            Boolean mask for small modes.
        """
        if not np.any(smallmode_idx):
            return

        zs = z[smallmode_idx]
        npts = self.etd_config.contour_points
        r = self.etd_config.contour_radius * np.exp(2j * np.pi * np.arange(0.5, npts) / npts)
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        def avg_phi(func, factor=1):
            return h * np.sum(func(factor * Z), axis=1) / npts

        psi1_14, psi2_14 = avg_phi(psi1, 1 / 4), avg_phi(psi2, 1 / 4)
        psi1_12, psi2_12 = avg_phi(psi1, 1 / 2), avg_phi(psi2, 1 / 2)
        psi1_34, psi2_34 = avg_phi(psi1, 3 / 4), avg_phi(psi2, 3 / 4)
        psi1_1, psi2_1, psi3_1 = avg_phi(psi1), avg_phi(psi2), avg_phi(psi3)

        idx = smallmode_idx
        self._a21[idx] = psi1_14 / 4.0
        self._a31[idx] = (psi1_14 - psi2_14 / 2.0) / 4.0
        self._a32[idx] = psi2_14 / 8.0
        self._a41[idx] = (psi1_12 - psi2_12) / 2.0
        self._a43[idx] = psi2_12 / 2.0
        self._a51[idx] = 3.0 * (psi1_34 - 3.0 * psi2_34 / 4.0) / 4.0
        self._a52[idx] = -3 * psi1_34 / 8.0
        self._a54[idx] = 9 * psi2_34 / 16.0
        self._a61[idx] = (-77 * psi1_1 + 59 * psi2_1) / 42.0
        self._a62[idx] = 8 * psi1_1 / 7.0
        self._a63[idx] = (111 * psi1_1 - 87 * psi2_1) / 28.0
        self._a65[idx] = (-47 * psi1_1 + 143 * psi2_1) / 84.0
        self._a71[idx] = 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        self._a73[idx] = (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._a74[idx] = 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._a75[idx] = (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._a76[idx] = (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

    def stage_init(self, u: np.ndarray) -> None:
        """
        Initialize nonlinear term for new step.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform one ETD35 stage pass for a diagonal system.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        if accept:
            self.stage_init(u)

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
    r"""
    ETD35 solver for non-diagonal systems via eigenvector diagonalization.

    Diagonalizes the full matrix :math:`\mathcal{L}` using eigen-decomposition:

    .. math::

        \mathcal{L} = \mathbf{S} \, \mathbf{\Lambda} \, \mathbf{S}^{-1},

    and performs all ETD(3,5) computations in the diagonal basis of
    :math:`\mathbf{\Lambda}`.

    Parameters
    ----------
    lin_op : np.ndarray
        Full (square) matrix linear operator :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    etd_config : ETDConfig
        ETD configuration object controlling spectral radius and contour settings.

    Notes
    -----
    This variant transforms the system into the eigenbasis:

    .. math::

        \mathbf{v} = \mathbf{S}^{-1}\mathbf{U}, \qquad
        \frac{d\mathbf{v}}{dt} = \mathbf{\Lambda}\mathbf{v}
            + \mathbf{S}^{-1}\mathcal{N}(\mathbf{S}\mathbf{v}).

    The ETD(3,5) coefficients are then computed elementwise in the diagonal
    eigenvalue space using the same :math:`\psi_k`-based update rules as
    the purely diagonal solver.

    Numerical stability requires the condition number of :math:`\mathbf{S}` to
    remain moderate (e.g. :math:`\kappa(\mathbf{S}) < 10^3`).
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initialize diagonalized ETD35 solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Full matrix linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        logger : logging.Logger, optional
            Logger instance for this solver.
        """
        super().__init__(lin_op, nl_func, etd_config, logger)
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
        """
        Update coefficients using eigenvalues.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self._eig_vals
        self._update_coeffs_diagonal(h, z)


    def stage_init(self, u: np.ndarray) -> None:
        """
        Initialize stored nonlinear evaluation and transformed state v = S^{-1} u.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform one ETD35 stage pass for diagonalized systems.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        # Use First is same as last principle (FSAL)
        if accept:
            self.stage_init(u)

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
    r"""
    ETD35 solver for full (non-diagonal, non-diagonalizable) linear operators.

    Implements the ETD(3,5) exponential time-differencing scheme directly
    using matrix exponentials and contour-integrated :math:`\psi` functions.

    Parameters
    ----------
    lin_op : np.ndarray
        Full matrix linear operator :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    etd_config : ETDConfig
        ETD configuration object (controls contour integration settings).

    Notes
    -----
    This version computes the matrix coefficients directly from

    .. math::

        \psi_k(\mathbf{Z}) =
            \frac{1}{(k-1)!} \int_0^1 e^{(1-\theta)\mathbf{Z}} \theta^{k-1}\,d\theta,

    where :math:`\mathbf{Z} = h\mathcal{L}`.

    To avoid ill-conditioning when :math:`\mathbf{Z}` has small eigenvalues,
    the integral is approximated by a complex contour average:

    .. math::

        \psi_k(\mathbf{Z}) \approx \frac{1}{N_p}\sum_{m=1}^{N_p}
            \psi_k(r_m)\,(\,r_m\mathbf{I} - \mathbf{Z}\,)^{-1},

    where :math:`r_m = R e^{2\pi i (m-1/2)/N_p}`.

    The stage updates follow the exponential Runge–Kutta pattern:

    .. math::

        \mathbf{k}_{i+1} =
            e^{c_i\mathbf{Z}}\mathbf{u}_n +
            \sum_j a_{ij}\,\mathcal{N}(\mathbf{k}_j),

    where :math:`a_{ij}` are constructed from combinations of
    :math:`\psi_1`, :math:`\psi_2`, and :math:`\psi_3`.

    This direct matrix form is slower but numerically robust for
    moderate-size dense systems where diagonalization is not feasible.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """
        Initialize ETD35 for non-diagonal systems.

        Parameters
        ----------
        lin_op : np.ndarray
            Full matrix linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        """
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
        """
        Update ETD coefficients for given step size h.

        Parameters
        ----------
        h : float
            Step size.
        """
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
        psi1_14, psi2_14, psi1_12, psi2_12, psi1_34, psi2_34 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(6)
        ]
        psi1_1, psi2_1, psi3_1 = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(3)]

        for point in contour_points:
            Q14 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 4.0)
            Q12 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 2.0)
            Q34 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - 3 * z / 4.0)
            Q = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z)
            psi1_14 += point * psi1(point) * Q14 / self.etd_config.contour_points
            psi2_14 += point * psi2(point) * Q14 / self.etd_config.contour_points
            psi1_12 += point * psi1(point) * Q12 / self.etd_config.contour_points
            psi2_12 += point * psi2(point) * Q12 / self.etd_config.contour_points
            psi1_34 += point * psi1(point) * Q34 / self.etd_config.contour_points
            psi2_34 += point * psi2(point) * Q34 / self.etd_config.contour_points
            psi1_1 += point * psi1(point) * Q / self.etd_config.contour_points
            psi2_1 += point * psi2(point) * Q / self.etd_config.contour_points
            psi3_1 += point * psi3(point) * Q / self.etd_config.contour_points

        self._a21 = h * psi1_14 / 4.0
        self._a31 = h * (psi1_14 - psi2_14 / 2.0) / 4.0
        self._a32 = h * psi2_14 / 8.0
        self._a41 = h * (psi1_12 - psi2_12) / 2.0
        self._a43 = h * psi2_12 / 2.0
        self._a51 = h * 3.0 * (psi1_34 - 3.0 * psi2_34 / 4.0) / 4.0
        self._a52 = -3 * h * psi1_34 / 8.0
        self._a54 = h * 9 * psi2_34 / 16.0
        self._a61 = h * (-77 * psi1_1 + 59 * psi2_1) / 42.0
        self._a62 = h * 8 * psi1_1 / 7.0
        self._a63 = h * (111 * psi1_1 - 87 * psi2_1) / 28.0
        self._a65 = h * (-47 * psi1_1 + 143 * psi2_1) / 84.0
        self._a71 = h * 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        self._a73 = h * (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._a74 = h * 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._a75 = h * (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._a76 = h * (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

    def stage_init(self, u: np.ndarray) -> None:
        """
        Initialize nonlinear term for new time step.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform one ETD35 Runge–Kutta update for non-diagonal systems.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        if accept:
            self.stage_init(u)

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
    r"""
    Adaptive fifth-order Exponential Time-Differencing Runge–Kutta solver (ETD(3,5)).

    Solves stiff systems of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` is a linear operator and :math:`\mathcal{N}` is a nonlinear term.

    This implementation follows the **ETD(3,5)** algorithm developed by
    Whalen, Brio, and Moloney (2015), which embeds a third-order ETD
    scheme for adaptive time-step control within a fifth-order integrator.

    The ETD(3,5) method advances the solution through exponential
    Runge-Kutta stages defined by the *ψ-functions*:

    .. math::

        \psi_r(z)
        = r \int_0^1 e^{(1-\theta)z}\,\theta^{r-1}\,d\theta,
        \quad r = 1,2,3,\dots

    These appear in the Runge-Kutta coefficients and can be related to
    the more common :math:`\phi`-functions via
    :math:`\psi_r(z) = r!\,\phi_r(z)`.

    The embedded third-order estimate is used to adapt the time step
    according to

    .. math::

        h_{\text{new}}
        = h_{\text{old}}\,\nu
            \left(
            \frac{\varepsilon}{\mathrm{err}}
            \right)^{1/(q+1)},
        \qquad q=4,

    where :math:`\varepsilon` is the tolerance, :math:`\nu` is a safety
    factor, and :math:`\mathrm{err}` is the estimated local truncation error.

    Supports
    --------
    - Diagonal :math:`\mathcal{L}` (elementwise exponentials)
    - Diagonalizable :math:`\mathcal{L}` (eigenbasis integration)
    - Full :math:`\mathcal{L}` (matrix exponentials via contour integration)

    References
    ----------
    Whalen, P., Brio, M., & Moloney, J. V. (2015).
    *Exponential time-differencing with embedded Runge–Kutta adaptive step control.*
    **Journal of Computational Physics**, 280, 579–601.
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
        """
        Initialize ETD35 adaptive solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator L in the system. May be a 1D array (diagonal system)
            or a 2D square matrix (non-diagonal system).
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function N(U).
        config : SolverConfig, optional
            General solver configuration controlling adaptivity thresholds,
            safety factors, and other integration parameters.
        etd_config : ETDConfig, optional
            Configuration for ETD-specific parameters, such as contour integration
            settings and spectral radius estimation.
        diagonalize : bool, optional
            If True, attempts eigenvalue decomposition to transform system into
            diagonal form before solving.
        loglevel : Union[str, int], optional
            Logging level.
        """
        super().__init__(
            lin_op,
            nl_func,
            config=config,
            etd_config=etd_config,
            loglevel=loglevel,
        )
        if self._diag:
            self._method = _Etd35Diagonal(lin_op, nl_func, etd_config, self.logger)
        else:
            if diagonalize:
                self._method = _Etd35Diagonalized(lin_op, nl_func, etd_config, self.logger)
            else:
                self._method = _Etd35NonDiagonal(lin_op, nl_func, etd_config)
        self._stages_init = False
        self._accept = False

    def _reset(self) -> None:
        """
        Reset internal solver state for restart or initialization.

        Resets cached coefficients and flags for adaptive stepping.
        """
        self._stages_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD35 coefficients if time step has changed.

        Parameters
        ----------
        h : float
            Time step size.
        """
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
            Current state vector at time step n.
        h : float
            Time step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The tuple (u_next, err_estimate) with the next-step value and
            the local truncation error estimate.
        """
        self._update_coeffs(h)
        if not self._stages_init:
            self._method.stage_init(u)
            self._stages_init = True
        return self._method.update_stages(u, self._accept)

    def _q(self) -> int:
        """
        Order variable for computing suggested step size (embedded order + 1).

        Returns
        -------
        int
            Effective order (4 for ETD(3,5)), used by the adaptive controller.
        """
        return 4
