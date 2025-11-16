r"""
Constant-Step Fifth-Order Exponential Time-Differencing Integrator
==================================================================

Implements a **fifth-order exponential time-differencing (ETD5) solver**
for stiff partial differential equations (PDEs) of the form

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
      = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

where :math:`\mathcal{L}` is a linear spatial differential operator
(e.g. Laplacian, biharmonic, etc.), and :math:`\mathcal{N}(\mathbf{U})`
is a nonlinear term in physical or spectral space.

The solver advances the field :math:`\mathbf{U}(x, t)` in time using
a **six-stage exponential Runge–Kutta (ETD5) method** with precomputed
matrix functions derived from :math:`\psi_r` integrals.

This constant-step version corresponds to the fixed-step form of the
adaptive ETD(3,5) scheme described in Whalen et al. (2015).

References
----------
Whalen, P., Brio, M., & Moloney, J.V. (2015).
*Exponential time-differencing with embedded Runge–Kutta adaptive step control.*
Journal of Computational Physics, 280, 579–601.
"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .etd import ETDCS, ETDConfig, psi1, psi2, psi3


class _Etd5Diagonal:  # pylint: disable=too-few-public-methods
    r"""
    ETD5 solver for diagonalized PDE systems.

    Solves evolution equations of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` acts diagonally in Fourier or spectral space.
    Each spectral mode evolves independently, allowing efficient
    elementwise computation of exponential and :math:`\psi_r` terms.

    The fifth-order ETD scheme uses six intermediate stages:

    .. math::

        \mathbf{U}(t + h)
          = e^{h\mathcal{L}}\,\mathbf{U}(t)
            + h \sum_{i=1}^{6} b_i(h\mathcal{L})\,\mathcal{N}_i,

    with stage updates

    .. math::

        \mathbf{k}_i
          = e^{c_i h\mathcal{L}}\,\mathbf{U}(t)
            + h \sum_{j<i} a_{ij}(h\mathcal{L})\,\mathcal{N}_j.

    The coefficients :math:`a_{ij}` and :math:`b_i` are derived from the
    **psi-functions**:

    .. math::

        \psi_r(z)
          = r \int_0^1 e^{(1-\theta)z}\,\theta^{r-1}\,d\theta,
          \quad r = 1,2,3,\dots

    For large :math:`|h\lambda|`, the functions are evaluated analytically.
    For small modes, they are computed by contour integration for
    numerical stability.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
    ) -> None:
        """Initialize ETD5 diagonal system strategy."""
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
        ) = [np.zeros(n, dtype=np.complex128) for _ in range(13)]
        self._b1, self._b3, self._b4, self._b5, self._b6 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """
        Update ETD5 coefficients based on step size h.

        Computes exponential and psi function coefficients for the diagonal system.
        Small modes (|`h*λ`| < modecutoff) use contour integration to avoid numerical
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
        psi1_14 = h * psi1(zb / 4)
        psi2_14 = h * psi2(zb / 4)
        psi1_12 = h * psi1(zb / 2)
        psi2_12 = h * psi2(zb / 2)
        psi1_34 = h * psi1(3 * zb / 4)
        psi2_34 = h * psi2(3 * zb / 4)
        psi1_1 = h * psi1(zb)
        psi2_1 = h * psi2(zb)
        psi3_1 = h * psi3(zb)

        self._a21[~smallmode_idx] = psi1_14 / 4.0
        self._a31[~smallmode_idx] = (psi1_14 - psi2_14 / 2.0) / 4.0
        self._a32[~smallmode_idx] = psi2_14 / 8.0
        self._a41[~smallmode_idx] = (psi1_12 - psi2_12) / 2.0
        self._a43[~smallmode_idx] = psi2_12 / 2.0
        self._a51[~smallmode_idx] = 3.0 * (psi1_34 - 3.0 * psi2_34 / 4.0) / 4.0
        self._a52[~smallmode_idx] = -3 * psi1_34 / 8.0
        self._a54[~smallmode_idx] = 9 * psi2_34 / 16.0
        self._a61[~smallmode_idx] = (-77 * psi1_1 + 59 * psi2_1) / 42.0
        self._a62[~smallmode_idx] = 8 * psi1_1 / 7.0
        self._a63[~smallmode_idx] = (111 * psi1_1 - 87 * psi2_1) / 28.0
        self._a65[~smallmode_idx] = (-47 * psi1_1 + 143 * psi2_1) / 84.0
        self._b1[~smallmode_idx] = 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        # Paper has error in b3 psi2 coefficient (states this is 497 but it is actually 467)
        self._b3[~smallmode_idx] = (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._b4[~smallmode_idx] = 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._b5[~smallmode_idx] = (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._b6[~smallmode_idx] = (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

        # compute small mode coeffs
        zs = z[smallmode_idx]  # z small
        r = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        psi1_14 = h * np.sum(psi1(Z / 4), axis=1) / self.etd_config.contour_points
        psi2_14 = h * np.sum(psi2(Z / 4), axis=1) / self.etd_config.contour_points
        psi1_12 = h * np.sum(psi1(Z / 2), axis=1) / self.etd_config.contour_points
        psi2_12 = h * np.sum(psi2(Z / 2), axis=1) / self.etd_config.contour_points
        psi1_34 = h * np.sum(psi1(3 * Z / 4), axis=1) / self.etd_config.contour_points
        psi2_34 = h * np.sum(psi2(3 * Z / 4), axis=1) / self.etd_config.contour_points
        psi1_1 = h * np.sum(psi1(Z), axis=1) / self.etd_config.contour_points
        psi2_1 = h * np.sum(psi2(Z), axis=1) / self.etd_config.contour_points
        psi3_1 = h * np.sum(psi3(Z), axis=1) / self.etd_config.contour_points

        self._a21[smallmode_idx] = psi1_14 / 4.0
        self._a31[smallmode_idx] = (psi1_14 - psi2_14 / 2.0) / 4.0
        self._a32[smallmode_idx] = psi2_14 / 8.0
        self._a41[smallmode_idx] = (psi1_12 - psi2_12) / 2.0
        self._a43[smallmode_idx] = psi2_12 / 2.0
        self._a51[smallmode_idx] = 3.0 * (psi1_34 - 3.0 * psi2_34 / 4.0) / 4.0
        self._a52[smallmode_idx] = -3 * psi1_34 / 8.0
        self._a54[smallmode_idx] = 9 * psi2_34 / 16.0
        self._a61[smallmode_idx] = (-77 * psi1_1 + 59 * psi2_1) / 42.0
        self._a62[smallmode_idx] = 8 * psi1_1 / 7.0
        self._a63[smallmode_idx] = (111 * psi1_1 - 87 * psi2_1) / 28.0
        self._a65[smallmode_idx] = (-47 * psi1_1 + 143 * psi2_1) / 84.0
        self._b1[smallmode_idx] = 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        # Paper has error in b3 psi2 coefficient (states this is 497 but it is actually 467)
        self._b3[smallmode_idx] = (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._b4[smallmode_idx] = 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._b5[smallmode_idx] = (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._b6[smallmode_idx] = (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

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
        self._NL1 = self.nl_func(self._k)  # FSAL Principle
        return self._k


class _Etd5NonDiagonal:  # pylint: disable=too-few-public-methods
    r"""
    ETD5 solver for non-diagonal PDE operators.

    Suitable for PDEs where the linear operator :math:`\mathcal{L}`
    couples multiple spatial modes, such as systems with mixed derivatives
    or nonlocal coupling.

    The governing equation is

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
          = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    and the solver advances :math:`\mathbf{U}` in time using the
    fifth-order ETD Runge–Kutta scheme with six exponential stages.

    The :math:`\psi_r(h\mathcal{L})` functions are defined as

    .. math::

        \psi_r(h\mathcal{L})
          = r \int_0^1 e^{(1-\theta)h\mathcal{L}}\,
            \theta^{r-1}\,d\theta,

    and are evaluated using **Cauchy contour integration** for small
    eigenvalue magnitudes:

    .. math::

        \psi_r(h\mathcal{L})
        \approx \frac{1}{N_p}
        \sum_{m=1}^{N_p}
        \psi_r(r_m)\,(r_m\mathbf{I} - h\mathcal{L})^{-1},
        \quad r_m = R\,e^{2\pi i(m - 1/2)/N_p}.

    This formulation is more computationally demanding than the diagonal
    variant but necessary for general coupled systems.

    References
    ----------
    Whalen, P., Brio, M., & Moloney, J.V. (2015).
    *Exponential time-differencing with embedded Runge–Kutta adaptive step control.*
    Journal of Computational Physics, 280, 579–601.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
    ) -> None:
        """
        Initialize ETD5 non-diagonal system strategy.

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
        ) = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(13)]
        self._b1, self._b3, self._b4, self._b5, self._b6 = [
            np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6 = [
            np.zeros(n, dtype=np.complex128) for _ in range(6)
        ]
        self._k = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """
        Update ETD5 coefficients for non-diagonal systems.

        Computes exponential and psi function coefficients using contour integration
        for all modes. Uses matrix exponentials and the Cauchy integral formula.

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
        self._b1 = h * 7 * (257 * psi1_1 - 497 * psi2_1 + 270 * psi3_1) / 2700
        self._b3 = h * (1097 * psi1_1 - 467 * psi2_1 - 150 * psi3_1) / 1350
        self._b4 = h * 2 * (-49 * psi1_1 + 199 * psi2_1 - 135 * psi3_1) / 225
        self._b5 = h * (-313 * psi1_1 + 883 * psi2_1 - 90 * psi3_1) / 1350
        self._b6 = h * (509 * psi1_1 - 2129 * psi2_1 + 1830 * psi3_1) / 2700

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize the first nonlinear evaluation (FSAL principle).

        Stores nl_func(u_n) for use in the first stage.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray) -> np.ndarray:
        """
        Advance solution by one time step using the six-stage ETD5 scheme.

        Executes the six Runge-Kutta-like stages of the ETD5 method using
        matrix-vector products for non-diagonal systems.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector at time step n.

        Returns
        -------
        np.ndarray
            Updated solution vector at time step n+1.
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
        self._NL1 = self.nl_func(self._k)  # FSAL principle
        return self._k


class ETD5(ETDCS):
    r"""
    Fifth-order Exponential Time-Differencing solver for PDEs.

    Integrates stiff PDEs of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
          = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U}),

    where
    - :math:`\mathcal{L}` is a linear spatial operator (e.g. diffusion or advection),
    - :math:`\mathcal{N}` represents nonlinear stiff terms.

    The solution is advanced in time using a **six-stage exponential Runge–Kutta**
    scheme, with coefficients derived from the :math:`\psi_r` functions.

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
    - Automatically detects whether the linear operator is diagonal and selects
      optimized routines.
    - Coefficients are cached and recomputed only when the step size :math:`h`
      changes.
    - Uses the *First Same As Last (FSAL)* property to reuse the last nonlinear
      evaluation efficiently.
    - Designed for PDEs discretized in space via spectral or finite-difference methods.

    References
    ----------
    Cox, S.M. & Matthews, P.C. (2002).
    *Exponential time differencing for stiff systems.*
    Journal of Computational Physics, 176(2), 430–455.

    Krogstad, S. (2005).
    *Generalized integrating factor methods for stiff PDEs.*
    Journal of Computational Physics, 203(1), 72–88.
    """

    _method: Union[_Etd5Diagonal, _Etd5NonDiagonal]

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
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
            If 1D array: treated as a diagonal operator (more efficient).
            If 2D array: treated as a full matrix operator.
            Supports both real and complex values.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function that maps the solution vector to its nonlinear contribution.
        etd_config : ETDConfig, default=ETDConfig()
            Configuration object containing modecutoff, contour_points, contour_radius.
        loglevel : Union[str, int], default="WARNING"
            Logging level.

        Notes
        -----
        - The solver automatically detects if lin_op is 1D (diagonal) and selects
          the optimized _Etd5Diagonal method; otherwise uses _Etd5NonDiagonal.
        - Internal state variables are initialized but coefficients are not computed
          until the first time step is taken.
        - The parent class ETDCS handles common setup and validation.
        """
        super().__init__(lin_op, nl_func, etd_config, loglevel)
        if self._diag:
            self._method = _Etd5Diagonal(lin_op, nl_func, self.etd_config)
        else:
            self._method = _Etd5NonDiagonal(lin_op, nl_func, self.etd_config)
        self.__n1_init = False

    def _reset(self) -> None:
        """
        Reset the solver to its initial state.

        Clears all cached coefficients and internal state variables, returning
        the solver to the state immediately after initialization. Useful when
        switching to a different initial condition or restarting a simulation.

        Notes
        -----
        - Clears the initialization flag, forcing reinitialization on next step.
        - Removes cached step size coefficients.
        - Does not affect the linear operator, nonlinear function, or configuration settings.
        """
        self.__n1_init = False
        self._h_coeff = None

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD5 coefficients if the step size has changed.

        Computes and caches the exponential and psi function coefficients required
        for the ETD5 method. Coefficients depend on h*L and are expensive to compute,
        so they are only recalculated when the step size changes.

        Parameters
        ----------
        h : float
            Time step size. Must be positive.

        Notes
        -----
        - If h equals the cached step size (self._h_coeff), returns immediately without recomputing.
        - Delegates actual computation to the internal method object (_Etd5Diagonal or _Etd5NonDiagonal).
        - Logs coefficient updates for debugging and monitoring.
        - The coefficients include matrix exponentials exp(h*L/4), exp(h*L/2),
          exp(3h*L/4), exp(h*L) and psi functions psi1, psi2, psi3 evaluated at various fractional steps.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD5 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance the solution by one time step using the ETD5 scheme.

        Computes u_{n+1} from u_n by executing the six-stage Runge-Kutta-like
        procedure of the ETD5 method. This includes evaluating the nonlinear
        function at intermediate stages and combining results using precomputed
        exponential and psi function coefficients.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector at time t_n.
        h : float
            Time step size (Δt = t_{n+1} - t_n). Must be positive.

        Returns
        -------
        np.ndarray
            Updated solution vector at time t_{n+1}.

        Notes
        -----
        - Automatically updates coefficients if h has changed since last call.
        - On the first call, initializes internal state by evaluating nl_func(u) and
          storing it for subsequent steps (FSAL principle).
        - The method uses six nonlinear function evaluations per step for 5th-order accuracy.
        - Delegates the actual stage computations to the internal method object.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u)
