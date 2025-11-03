r"""
Exponential Time-Differencing (ETD) Runge-Kutta Methods
============================================================================

Implements Exponential Time-Differencing (ETD) Runge-Kutta methods for solving
**stiff partial differential equations (PDEs)** of the form

.. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U})

where :math:`\mathcal{L}` is a stiff linear operator and
:math:`\mathcal{N}(\mathbf{U})` is a nonlinear, typically non-stiff term.

These methods use analytic integration of the linear part via the matrix exponential
and handle the nonlinear part with Runge-Kutta-like stages built around the
**psi-functions**:

.. math::

    \psi_r(z)
      = r \int_0^1 e^{(1-\theta)z}\,\theta^{r-1}\,d\theta,
      \quad r = 1,2,3,\dots

Contents
--------
- :class:`ETDConfig` - configuration for contour integration and cutoff settings.
- :func:`psi1`, :func:`psi2`, :func:`psi3` - exponential helper functions.
- :class:`ETDAS` - adaptive-step ETD solver.
- :class:`ETDCS` - constant-step ETD solver.

Notes
-----
ETD methods are particularly efficient when
:math:`\mathcal{N}(\mathbf{U})` evolves on a slower time scale than the linear component :math:`\mathcal{L}\mathbf{U}`.
"""

from typing import Union, Literal, Callable
import numpy as np
from .solvercs import BaseSolverCS
from .solveras import SolverConfig, BaseSolverAS


class ETDConfig:
    r"""
    Configuration parameters for Exponential Time-Differencing (ETD) solvers.

    These parameters control the numerical evaluation of the **psi-functions**
    and the stability of contour-integral approximations.

    Parameters
    ----------
    modecutoff : float, optional
        Threshold for switching between direct evaluation and contour integration.
        For :math:`|z| < \text{modecutoff}`, :math:`\psi_k(z)` is computed via
        Contour integration; otherwise, direct evaluation is used.
    contour_points : int, optional
        Number of quadrature nodes used for contour integration.
    contour_radius : float, optional
        Radius :math:`R` of the circular contour in the complex plane used for the
        Cauchy integral evaluation of :math:`\psi_k(z)`.

    Notes
    -----
    For small :math:`z`, the ETD psi-functions are numerically unstable via direct evaluation and a
    contour integral representation is preferred.

    .. math::

        \psi_k(z)
        = \frac{1}{2\pi i}
        \oint_\Gamma
        \frac{e^\xi}{(\xi - z)\xi^{k-1}}\,d\xi,
        \quad \Gamma : |\xi| = R.

    For larger :math:`z` (not near zero) direct evaluation of the ETD psi-functions works well.
    """

    def __init__(self, modecutoff: float = 0.01, contour_points: int = 32, contour_radius: float = 1.0) -> None:
        self._modecutoff: float = 0.01
        self._contour_points: int = 32
        self._contour_radius: float = 1.0
        self.modecutoff = modecutoff
        self.contour_points = contour_points
        self.contour_radius = contour_radius

    @property
    def modecutoff(self) -> float:
        """Numerical inaccuracy cutoff for psi functions.

        Must be between 0.0 and 1.0.
        """
        return self._modecutoff

    @modecutoff.setter
    def modecutoff(self, value: float) -> None:
        if (value > 1.0) or (value <= 0):
            raise ValueError(f"modecutoff must be between 0.0 and 1.0 but is {value}")
        self._modecutoff = value

    @property
    def contour_points(self) -> int:
        """Number of points for the contour integral.

        Must be an integer greater than 1.
        """
        return self._contour_points

    @contour_points.setter
    def contour_points(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"contour_points must be an integer but is {value}")
        if value <= 1:
            raise ValueError(f"contour_points must be an integer greater than 1 but is {value}")
        self._contour_points = value

    @property
    def contour_radius(self) -> float:
        """Radius of the circular contour integral.

        Must be greater than 0.
        """
        return self._contour_radius

    @contour_radius.setter
    def contour_radius(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"contour_radius must greater than 0 but is {value}")
        self._contour_radius = value


def psi1(z: np.ndarray) -> np.ndarray:
    r"""
    Compute :math:`\psi_1(z) = \frac{e^z - 1}{z}` element-wise.

    Parameters
    ----------
    z : np.ndarray
        Real or complex array.

    Returns
    -------
    np.ndarray
        Array of the same shape as `z`.
    """
    return (np.exp(z) - 1) / z


def psi2(z: np.ndarray) -> np.ndarray:
    r"""
    Compute :math:`\psi_2(z) = 2*\frac{e^z - 1 - z}{z^2}` element-wise.

    Parameters
    ----------
    z : np.ndarray
        Real or complex array.

    Returns
    -------
    np.ndarray
        Array of the same shape as `z`.
    """
    return 2 * (np.exp(z) - 1 - z) / z**2


def psi3(z: np.ndarray) -> np.ndarray:
    r"""
    Compute :math:`\psi_3(z) = 6*\frac{e^z - 1 - z - \frac{z^2}{2}}{z^3}` element-wise.

    Parameters
    ----------
    z : np.ndarray
        Real or complex array.

    Returns
    -------
    np.ndarray
        Array of the same shape as `z`.
    """
    return 6 * (np.exp(z) - 1 - z - z**2 / 2) / z**3


class ETDAS(BaseSolverAS):
    r"""
    Adaptive-step Exponential Time-Differencing (ETD) Runge–Kutta solver.

    Integrates stiff PDE's of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U})

    where :math:`\mathcal{L}` is a linear operator and :math:`\mathcal{N}` is nonlinear.

    This variant adjusts the time step :math:`h` dynamically based on embedded
    error estimates, maintaining both accuracy and efficiency.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    config : SolverConfig, optional
        General solver configuration controlling adaptivity.
    etd_config : ETDConfig, optional
        ETD-specific contour and cutoff settings.
    loglevel : str or int, optional
        Logging verbosity level.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        super().__init__(lin_op, nl_func, config, loglevel=loglevel)
        self.etd_config: ETDConfig = etd_config
        self._h_coeff = None


class ETDCS(BaseSolverCS):
    r"""
    Constant-step Exponential Time-Differencing (ETD) Runge–Kutta solver.

    Integrates stiff PDEs of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U})

    using a **fixed step size** :math:`h`.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    etd_config : ETDConfig, optional
        ETD configuration settings.
    loglevel : str or int, optional
        Logging level.

    Notes
    -----
    Constant-step ETD methods are efficient when the stiffness and smoothness of
    the system are known in advance.  Each time step advances via

    .. math::

        \mathbf{u}_{n+1}
        = e^{h\mathcal{L}}\mathbf{u}_n
            + h\sum_i b_i\,\mathcal{N}(\mathbf{k}_i),

    where :math:`\mathbf{k}_i` are stage vectors computed using
    matrix–function coefficients :math:`\psi_k(h\mathcal{L})`.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        super().__init__(lin_op, nl_func, loglevel=loglevel)
        self.etd_config: ETDConfig = etd_config
        self._h_coeff = None
