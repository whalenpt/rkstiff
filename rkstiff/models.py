"""
rkstiff.models

Provides common models and initial conditions for testing stiff solvers.

This module includes:
    - KdV (Korteweg-de Vries) equation utilities
    - Burgers equation utilities
    - Allen-Cahn equation utilities
"""

from typing import Callable, Tuple, Sequence
import numpy as np


def kdv_soliton(x: np.ndarray, ampl: float = 0.5, x0: float = 0.0, t: float = 0.0) -> np.ndarray:
    """
    Return a single KdV soliton initial condition.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinate array.
    ampl : float, optional
        Amplitude parameter of the soliton. Default is 0.5.
    x0 : float, optional
        Initial position of the soliton. Default is 0.0.
    t : float, optional
        Time parameter (for time-evolved soliton). Default is 0.0.

    Returns
    -------
    np.ndarray
        Soliton solution evaluated at positions x and time t.
    """
    u0 = 0.5 * ampl**2 / (np.cosh(ampl * (x - x0 - ampl**2 * t) / 2) ** 2)
    return u0


def kdv_multi_soliton(x: np.ndarray, ampl: Sequence[float], x0: Sequence[float], t: float = 0.0) -> np.ndarray:
    """
    Return a multi-soliton KdV initial condition.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinate array of length n.
    ampl : Sequence[float]
        Amplitude parameters for each soliton (length m).
    x0 : Sequence[float]
        Initial positions for each soliton (length m).
    t : float, optional
        Time parameter (for time-evolved solitons). Default is 0.0.

    Returns
    -------
    np.ndarray
        Sum of all soliton solutions evaluated at positions x and time t.

    Raises
    ------
    ValueError
        If lengths of ampl and x0 don't match.
    """
    if len(x0) != len(ampl):
        raise ValueError(f"Length of ampl ({len(ampl)}) must equal length of x0 ({len(x0)})")

    m = len(ampl)
    n = len(x)
    ampl_arr = np.array(ampl).reshape(1, m)
    x0_arr = np.array(x0).reshape(1, m)

    u0 = 0.5 * ampl_arr**2 / (np.cosh(ampl_arr * (x.reshape(n, 1) - x0_arr - ampl_arr**2 * t) / 2) ** 2)
    u0 = np.sum(u0, axis=1)

    return u0


def kdv_ops(kx: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Return the linear operator and nonlinear function for the KdV equation.

    The KdV equation is: u_t + 6u*u_x + u_xxx = 0

    Parameters
    ----------
    kx : np.ndarray
        Wavenumber array in Fourier space.

    Returns
    -------
    lin_op : np.ndarray
        Linear operator in Fourier space (1j * kx**3).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function that computes -6 * FFT(u * u_x).
    """
    lin_op = 1j * kx**3

    def nl_func(uf: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -6 * np.fft.rfft(u * ux)

    return lin_op, nl_func


def burgers_ops(kx: np.ndarray, mu: float) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Return the linear operator and nonlinear function for the viscous Burgers equation.

    The Burgers equation is: u_t + u*u_x = mu * u_xx

    Parameters
    ----------
    kx : np.ndarray
        Wavenumber array in Fourier space.
    mu : float
        Viscosity parameter (must be positive).

    Returns
    -------
    lin_op : np.ndarray
        Linear operator in Fourier space (-mu * kx**2).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function that computes -FFT(u * u_x).
    """
    lin_op = -mu * kx**2

    def nl_func(uf: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -np.fft.rfft(u * ux)

    return lin_op, nl_func


def allen_cahn_ops(
    x: np.ndarray, d_cheb_matrix: np.ndarray, epsilon: float = 0.01
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Return the linear operator and nonlinear function for the Allen-Cahn equation.

    The Allen-Cahn equation is: u_t = epsilon * u_xx + u - u^3

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinate array (Chebyshev grid).
    d_cheb_matrix : np.ndarray
        Chebyshev differentiation matrix.
    epsilon : float, optional
        Diffusion coefficient (must be positive). Default is 0.01.

    Returns
    -------
    lin_op : np.ndarray
        Linear operator (epsilon * D^2 + I) with boundary points removed.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function that computes x - (u + x)^3.

    Notes
    -----
    Boundary points (first and last) are removed from the linear operator,
    as they are typically treated separately in boundary value problems.
    """
    d2_cheb_matrix = d_cheb_matrix.dot(d_cheb_matrix)
    lin_op = epsilon * d2_cheb_matrix + np.eye(*d2_cheb_matrix.shape)
    lin_op = lin_op[1:-1, 1:-1]  # Remove boundary points

    def nl_func(u: np.ndarray) -> np.ndarray:
        return x[1:-1] - np.power(u + x[1:-1], 3)

    return lin_op, nl_func
