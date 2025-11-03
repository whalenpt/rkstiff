r"""
Grid construction utilities for spectral and transform methods
==============================================================


This module provides functions for constructing **spatial** and **spectral**
grids used in Fourier, Chebyshev, and Hankel spectral methods.
These grids form the basis of spatial discretizations for PDE solvers
such as those implemented in :mod:`rkstiff`.

Overview
--------

Each function constructs either:

- A **uniform grid** for FFT-based spectral differentiation.
- A **Chebyshev–Gauss–Lobatto** grid for non-periodic domains.
- A **radial grid** for axisymmetric problems using the Hankel transform.

All grids and wavenumber sets are compatible with NumPy and SciPy FFTs
and can be directly used with the derivative utilities in
:mod:`rkstiff.derivatives`.

Contents
--------

- :func:`construct_x_kx_rfft` — Uniform grid for rFFT (real-valued FFT)
- :func:`construct_x_kx_fft` — Uniform grid for FFT (complex-valued)
- :func:`construct_x_cheb` — Chebyshev spatial grid
- :func:`construct_x_dx_cheb` — Chebyshev grid with differentiation matrix
- :func:`construct_r_kr_hankel` — Radial grid for Hankel transforms
- :func:`mirror_grid` — Symmetric grid reflection utility
"""

from typing import Optional, Tuple
import numpy as np
import scipy.special as sp  # type: ignore


def construct_x_kx_rfft(n: int, a: float = 0.0, b: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construct a uniform 1D spatial grid and *rFFT-compatible* wavenumber grid.

    Parameters
    ----------
    n : int
        Number of grid points. Must be an even integer greater than 2.
    a : float, optional
        Left endpoint of the spatial domain. Default is ``0.0``.
    b : float, optional
        Right endpoint of the spatial domain. Default is ``2π``.

    Returns
    -------
    x : np.ndarray
        Uniform spatial grid with ``n`` points in :math:`[a, b)`.
    kx : np.ndarray
        Spectral wavenumber grid with ``n/2 + 1`` points (for use with rFFT).

    Notes
    -----
    The grid spacing and wavenumbers are given by:

    .. math::

        \Delta x = \frac{b - a}{n}, \qquad
        k_x = 2\pi \, \mathrm{rfftfreq}(n, \Delta x).

    Examples
    --------
    >>> x, kx = construct_x_kx_rfft(128, a=0, b=2*np.pi)
    >>> x.shape
    (128,)
    >>> kx.shape
    (65,)
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n <= 2 or (n % 2) != 0:
        raise ValueError("n must be an even integer greater than 2.")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    return x, kx


def construct_x_kx_fft(n: int, a: float = 0.0, b: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construct a uniform 1D spatial grid and *FFT-compatible* wavenumber grid.

    Parameters
    ----------
    n : int
        Number of grid points. Must be an even integer greater than 2.
    a : float, optional
        Left endpoint of the spatial domain. Default is ``0.0``.
    b : float, optional
        Right endpoint of the spatial domain. Default is ``2π``.

    Returns
    -------
    x : np.ndarray
        Uniform spatial grid with ``n`` points in :math:`[a, b)`.
    kx : np.ndarray
        Spectral wavenumber grid with ``n`` points (for use with FFT).

    Notes
    -----
    The uniform grid and Fourier frequencies are defined by:

    .. math::

        \Delta x = \frac{b - a}{n}, \qquad
        k_x = 2\pi \, \mathrm{fftfreq}(n, \Delta x).

    Examples
    --------
    >>> x, kx = construct_x_kx_fft(128)
    >>> x.shape, kx.shape
    ((128,), (128,))
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n <= 2 or (n % 2) != 0:
        raise ValueError("n must be an even integer greater than 2.")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return x, kx


def construct_x_cheb(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""
    Construct a 1D grid with Chebyshev–Gauss–Lobatto points.

    Parameters
    ----------
    n : int
        Polynomial order (number of subintervals). The grid has ``n + 1`` points.
    a, b : float, optional
        Interval endpoints. Defaults are ``a = -1``, ``b = 1``.

    Returns
    -------
    np.ndarray
        Grid of ``n + 1`` Chebyshev points mapped to the interval :math:`[a, b]`.

    Notes
    -----
    The Chebyshev points on :math:`[-1, 1]` are given by:

    .. math::

        x_j = \cos\!\left(\frac{j\pi}{n}\right),
        \quad j = 0, 1, \dots, n,

    which are then linearly mapped to :math:`[a, b]`.

    Examples
    --------
    >>> x = construct_x_cheb(10, a=-1, b=1)
    >>> len(x)
    11
    >>> x[0], x[-1]
    (1.0, -1.0)
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n < 2:
        raise ValueError("n must be ≥ 2.")
    x = np.polynomial.chebyshev.chebpts2(n + 1)
    x = a + (b - a) * (x + 1) / 2.0
    return x


def construct_x_dx_cheb(n: int, a: float = -1, b: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construct Chebyshev–Gauss–Lobatto grid and its differentiation matrix.

    Parameters
    ----------
    n : int
        Polynomial order (number of subintervals). Produces ``n + 1`` points.
    a, b : float, optional
        Interval endpoints. Defaults are ``a = -1``, ``b = 1``.

    Returns
    -------
    x : np.ndarray
        Chebyshev grid points on :math:`[a, b]`.
    d_cheb_matrix : np.ndarray, shape (n+1, n+1)
        Differentiation matrix :math:`D` such that ``D @ f ≈ df/dx``.

    Notes
    -----
    The entries of the differentiation matrix are:

    .. math::

        D_{ij} =
        \begin{cases}
            \dfrac{c_i}{c_j (x_i - x_j)}, & i \neq j, \\
            -\sum_{k \neq i} D_{ik}, & i = j,
        \end{cases}

    where :math:`c_i = 2(-1)^i` for endpoints and :math:`c_i = (-1)^i` otherwise.

    The matrix achieves spectral accuracy for smooth functions
    and satisfies the property :math:`D\mathbf{1} = 0`.
    """
    x = construct_x_cheb(n, a, b)
    c = np.r_[2, np.ones(n - 1), 2] * np.power(-1, np.arange(0, n + 1))
    X = np.tile(x.reshape(n + 1, 1), (1, n + 1))
    dX = X - X.T
    d_cheb_matrix = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    d_cheb_matrix = d_cheb_matrix - np.diag(d_cheb_matrix.sum(axis=1))
    return x, d_cheb_matrix


def construct_r_kr_hankel(nr: int, rmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""
    Construct Hankel transform grids for axisymmetric domains.

    Parameters
    ----------
    nr : int
        Number of radial grid points (≥ 4).
    rmax : float
        Maximum radius of the domain.

    Returns
    -------
    r : np.ndarray
        Radial grid points in :math:`(0, r_\max]`.
    kr : np.ndarray
        Spectral grid points in wavenumber space.
    bessel_zeros : np.ndarray
        First ``nr`` zeros of :math:`J_0`.
    jN : float
        The ``(nr+1)``-th zero of :math:`J_0` used for normalization.

    Notes
    -----
    The grid is defined from the zeros of the Bessel function :math:`J_0`:

    .. math::

        r_i = \frac{j_i}{j_{N+1}} \, r_\max, \qquad
        k_i = \frac{j_i}{r_\max},

    where :math:`j_i` are the zeros of :math:`J_0`.
    """
    if nr < 4:
        raise ValueError("nr must be ≥ 4.")
    if rmax <= 0:
        raise ValueError("rmax must be positive.")

    bessel_zeros = sp.jn_zeros(0, nr + 1)
    bessel_zeros, jN = bessel_zeros[:-1], bessel_zeros[-1]
    r = bessel_zeros * rmax / jN
    kr = bessel_zeros / rmax
    return r, kr, bessel_zeros, jN


def mirror_grid(
    r: np.ndarray, u: Optional[np.ndarray] = None, axis: int = -1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""
    Mirror a radial grid (and optional function) to produce a symmetric domain.

    Parameters
    ----------
    r : np.ndarray
        Radial grid on :math:`[0, r_\max]`.
    u : np.ndarray, optional
        Function values at ``r``. If provided, mirrored values are returned.
    axis : int, optional
        Axis along which to mirror ``u``. Default is ``-1``.

        * ``-1`` — Stack horizontally (for 1D)
        * ``0`` — Stack vertically (rows)
        * ``1`` — Stack horizontally (columns)

    Returns
    -------
    rnew : np.ndarray
        Mirrored grid on :math:`[-r_\max, r_\max]`.
    unew : np.ndarray, optional
        Mirrored function values (if ``u`` was provided).

    Notes
    -----
    Useful for visualizing radially symmetric solutions or enforcing
    symmetric boundary conditions.

    Examples
    --------
    >>> r = np.array([0, 1, 2, 3])
    >>> u = np.array([1, 2, 3, 4])
    >>> rnew, unew = mirror_grid(r, u)
    >>> rnew
    array([-3, -2, -1, 0, 0, 1, 2, 3])
    >>> unew
    array([4, 3, 2, 1, 1, 2, 3, 4])
    """
    rnew = np.hstack([-np.flipud(r), r])
    if u is None:
        return rnew, None

    if axis == -1:
        unew = np.hstack([np.flipud(u), u])
    elif axis == 0:
        unew = np.vstack([np.flipud(u), u])
    elif axis == 1:
        unew = np.hstack([np.fliplr(u), u])
    else:
        raise ValueError("axis must be -1, 0, or 1.")

    return rnew, unew
