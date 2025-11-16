r"""
Common benchmark models and initial conditions for stiff PDE solvers
====================================================================


This module provides utilities for constructing canonical **test equations**
used to validate time-integration schemes such as exponential, integrating-factor,
and Runge–Kutta methods.

Each model supplies:
  - The governing **PDE**
  - The corresponding **linear** and **nonlinear operators**
  - Representative **initial conditions**

Models included
---------------

- **Korteweg–de Vries (KdV)** equation — soliton dynamics
- **Viscous Burgers** equation — nonlinear advection–diffusion
- **Allen–Cahn** equation — bistable phase-field dynamics
"""

from typing import Callable, Tuple, Sequence
import numpy as np


# ---------------------------------------------------------------------------
#  KORTEWEG–DE VRIES EQUATION
# ---------------------------------------------------------------------------


def kdv_soliton(x: np.ndarray, ampl: float = 0.5, x0: float = 0.0, t: float = 0.0) -> np.ndarray:
    r"""
    Return the analytic single-soliton solution of the **Korteweg–de Vries (KdV)** equation.

    The KdV equation is:

    .. math::

        \frac{\partial u}{\partial t}
        + 6 u \frac{\partial u}{\partial x}
        + \frac{\partial^3 u}{\partial x^3} = 0.

    Its single-soliton solution is given by:

    .. math::

        u(x,t)
        = \tfrac{1}{2} a^2
          \operatorname{sech}^2\!
          \left[\tfrac{1}{2} a
          (x - x_0 - a^2 t)\right],

    where :math:`a` is the amplitude parameter.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid points.
    ampl : float, optional
        Soliton amplitude :math:`a`. Default is ``0.5``.
    x0 : float, optional
        Initial position :math:`x_0`. Default is ``0.0``.
    t : float, optional
        Time. Default is ``0.0``.

    Returns
    -------
    np.ndarray
        Soliton profile :math:`u(x,t)`.
    """
    u0 = 0.5 * ampl**2 / (np.cosh(ampl * (x - x0 - ampl**2 * t) / 2) ** 2)
    return u0


def kdv_multi_soliton(x: np.ndarray, ampl: Sequence[float], x0: Sequence[float], t: float = 0.0) -> np.ndarray:
    r"""
    Construct a **multi-soliton** superposition for the KdV equation.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid points.
    ampl : Sequence[float]
        Sequence of soliton amplitudes :math:`(a_1, a_2, \dots, a_m)`.
    x0 : Sequence[float]
        Initial positions :math:`(x_{0,1}, x_{0,2}, \dots, x_{0,m})`.
    t : float, optional
        Time. Default is ``0.0``.

    Returns
    -------
    np.ndarray
        Sum of all soliton profiles at time ``t``.

    Raises
    ------
    ValueError
        If ``ampl`` and ``x0`` have mismatched lengths.
    """
    if len(x0) != len(ampl):
        raise ValueError("Lengths of ampl and x0 must match.")

    m = len(ampl)
    n = len(x)
    ampl_arr = np.array(ampl).reshape(1, m)
    x0_arr = np.array(x0).reshape(1, m)

    u0 = 0.5 * ampl_arr**2 / (np.cosh(ampl_arr * (x.reshape(n, 1) - x0_arr - ampl_arr**2 * t) / 2) ** 2)
    return np.sum(u0, axis=1)


def kdv_ops(kx: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    r"""
    Return the linear and nonlinear operators for the **KdV** equation.

    Governing PDE:

    .. math::

        \frac{\partial u}{\partial t}
        = -6u \frac{\partial u}{\partial x}
          - \frac{\partial^3 u}{\partial x^3}.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumber array in Fourier space.

    Returns
    -------
    lin_op : np.ndarray
        Linear operator in spectral space :math:`L = i k_x^3`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear term :math:`N(u) = -6\,\mathcal{F}[u u_x]`,
        where :math:`u_x = \mathcal{F}^{-1}[i k_x \hat{u}]`.
    """
    lin_op = 1j * kx**3

    def nl_func(uf: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -6 * np.fft.rfft(u * ux)

    return lin_op, nl_func


# ---------------------------------------------------------------------------
#  VISCOUS BURGERS EQUATION
# ---------------------------------------------------------------------------


def burgers_ops(kx: np.ndarray, mu: float) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    r"""
    Return the linear and nonlinear operators for the **viscous Burgers equation**.

    Governing PDE:

    .. math::

        \frac{\partial u}{\partial t}
        + u \frac{\partial u}{\partial x}
        = \mu \frac{\partial^2 u}{\partial x^2},

    where :math:`\mu > 0` is the kinematic viscosity.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumber array in Fourier space.
    mu : float
        Viscosity coefficient.

    Returns
    -------
    lin_op : np.ndarray
        Linear operator :math:`L = -\mu k_x^2`.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear term :math:`N(u) = -\mathcal{F}[u u_x]`.

    Notes
    -----
    In Fourier space, differentiation corresponds to multiplication by
    :math:`i k_x`, allowing both linear diffusion and nonlinear advection
    terms to be computed spectrally.
    """
    lin_op = -mu * kx**2

    def nl_func(uf: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -np.fft.rfft(u * ux)

    return lin_op, nl_func


# ---------------------------------------------------------------------------
#  ALLEN–CAHN EQUATION
# ---------------------------------------------------------------------------


def allen_cahn_ops(
    x: np.ndarray, d_cheb_matrix: np.ndarray, epsilon: float = 0.01
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    r"""
    Return the linear and nonlinear operators for the **Allen–Cahn equation**.

    Governing PDE:

    .. math::

        \frac{\partial u}{\partial t}
        = \epsilon \frac{\partial^2 u}{\partial x^2}
        + u - u^3,

    where :math:`\epsilon` is a small positive diffusion coefficient.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid (Chebyshev–Gauss–Lobatto points).
    d_cheb_matrix : np.ndarray
        Chebyshev differentiation matrix :math:`D`.
    epsilon : float, optional
        Diffusion parameter :math:`\epsilon`. Default is ``0.01``.

    Returns
    -------
    lin_op : np.ndarray
        Linear operator :math:`L = \epsilon D^2 + I` with boundary rows removed.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear term :math:`N(u) = u - u^3`.

    Notes
    -----
    Boundary points are excluded from ``lin_op`` since Dirichlet or
    Neumann boundary conditions are typically enforced separately.
    """
    d2_cheb_matrix = d_cheb_matrix.dot(d_cheb_matrix)
    lin_op = epsilon * d2_cheb_matrix + np.eye(*d2_cheb_matrix.shape)
    lin_op = lin_op[1:-1, 1:-1]

    def nl_func(u: np.ndarray) -> np.ndarray:
        val = x[1:-1] - np.power(u + x[1:-1], 3)
        return np.asarray(val, dtype=np.complex128).ravel()

    return lin_op, nl_func
