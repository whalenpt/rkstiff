r"""
rkstiff.derivatives
===================

Spectral differentiation utilities for Fourier and Chebyshev grids.

This module provides high-accuracy derivative operators using spectral methods.
All functions are compatible with NumPy arrays and designed for use in
stiff ODE/PDE solvers.

Functions
---------

- **dx_rfft** — Derivatives using the *real* FFT (rFFT)
- **dx_fft** — Derivatives using the *complex* FFT (FFT)
- **dx_cheb** — Derivatives using the Chebyshev spectral differentiation matrix

Each method computes derivatives according to:

.. math::

    \frac{d^n u}{dx^n} = \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right]

for Fourier methods, or

.. math::

    \frac{d^n u}{dx^n} = D^n u

for Chebyshev methods, where :math:`D` is the Chebyshev differentiation matrix.
"""

import numpy as np


def dx_rfft(kx: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative of a real-valued array using the real FFT (rFFT).

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid. For input size :math:`N`, ``kx`` must
        have size :math:`N/2 + 1`. Typically generated using
        ``np.fft.rfftfreq(N, d=dx)``.
    u : np.ndarray
        Real-valued input array. Must **not** be complex.
    n : int, optional
        Derivative order (non-negative integer). Default is ``1``.

    Returns
    -------
    np.ndarray
        The *n*-th derivative of ``u``, same shape as the input.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or if ``u`` is complex-valued.
    ValueError
        If ``n`` is negative or ``kx`` has an incompatible shape.

    Notes
    -----
    For a real FFT, if the input array has size :math:`N`, the rFFT output
    has size :math:`N/2 + 1`. Therefore, ``kx`` must match that size.

    The derivative is computed as

    .. math::

        \frac{d^n u}{dx^n} =
        \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right].

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.sin(x)
    >>> dudx = dx_rfft(kx, u, n=1)
    >>> np.allclose(dudx, np.cos(x), atol=1e-10)
    True
    """
    if not isinstance(n, int):
        raise TypeError(f"derivative order n must be an integer, it is {n}")
    if n < 0:
        raise ValueError(f"derivative order n must be non-negative, it is {n}")

    # Check if u is complex-valued
    if np.iscomplexobj(u):
        raise TypeError("dx_rfft requires real-valued input array. For complex arrays, use dx_fft instead.")

    if u.size == 0:
        return np.array([], dtype=float)

    if n == 0:
        return u

    u_fft = np.fft.rfft(u)

    # Validate shape compatibility
    if kx.shape != u_fft.shape:
        expected_kx_size = u_fft.shape[-1]
        raise ValueError(
            f"Shape mismatch: kx has shape {kx.shape} but should have shape "
            f"matching rfft output {u_fft.shape}. For input size {u.shape[-1]}, "
            f"kx should have size {expected_kx_size} (N//2 + 1)."
        )

    if n == 1:
        uxn = np.fft.irfft(1j * kx * u_fft, n=u.shape[-1])
    else:
        uxn = np.fft.irfft(np.power(1j * kx, n) * u_fft, n=u.shape[-1])
    return uxn


def dx_fft(kx: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative of a complex-valued array using FFT.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid. Must have the same size as ``u``.
    u : np.ndarray
        Complex-valued input array.
    n : int, optional
        Derivative order (non-negative integer). Default is ``1``.

    Returns
    -------
    np.ndarray
        The *n*-th derivative of ``u``.

    Notes
    -----
    This implements the Fourier differentiation rule:

    .. math::

        \frac{d^n u}{dx^n} =
        \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right].

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.exp(1j * x)
    >>> dudx = dx_fft(kx, u, n=1)
    >>> np.allclose(dudx, 1j * np.exp(1j * x), atol=1e-10)
    True
    """
    if not isinstance(n, int):
        raise TypeError(f"derivative order n must be an integer, it is {n}")
    if n < 0:
        raise ValueError(f"derivative order n must be non-negative, it is {n}")

    if n == 0:
        return u

    u_fft = np.fft.fft(u)

    # Validate shape compatibility
    if kx.shape != u_fft.shape:
        raise ValueError(
            f"Shape mismatch: kx has shape {kx.shape} but should match " f"u shape {u.shape} for complex FFT."
        )

    if n == 1:
        uxn = np.fft.ifft(1j * kx * u_fft)
    else:
        uxn = np.fft.ifft(np.power(1j * kx, n) * u_fft)
    return uxn


def dx_cheb(d_cheb_matrix: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative using a Chebyshev spectral differentiation matrix.

    Parameters
    ----------
    d_cheb_matrix : np.ndarray
        Chebyshev differentiation matrix :math:`D \in \mathbb{R}^{N\times N}`,
        typically generated using ``construct_x_dx_cheb``.
    u : np.ndarray
        Function values sampled on the Chebyshev–Gauss–Lobatto grid.
        Shape ``(N,)`` or ``(N, M)``.
    n : int, optional
        Derivative order (non-negative integer). Default is ``1``.

    Returns
    -------
    np.ndarray
        The *n*-th derivative of ``u``, same shape as the input.

    Notes
    -----
    The derivative is computed by repeated matrix multiplication:

    .. math::

        \frac{du}{dx} = D u, \quad
        \frac{d^n u}{dx^n} = D^n u.

    This method achieves *spectral accuracy* for smooth functions but may
    lose numerical stability for large :math:`n` due to powers of :math:`D`.

    For a function defined on :math:`[a,b]`, the matrix must be scaled by
    :math:`2/(b-a)` to account for domain mapping from :math:`[-1,1]`.

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.grids import construct_x_dx_cheb
    >>> x, D = construct_x_dx_cheb(n=20, a=-1, b=1)
    >>> u = np.sin(x)
    >>> dudx = dx_cheb(D, u, n=1)
    >>> np.allclose(dudx, np.cos(x), atol=1e-10)
    True
    >>> d2udx2 = dx_cheb(D, u, n=2)
    >>> np.allclose(d2udx2, -np.sin(x), atol=1e-8)
    True
    """

    if not isinstance(n, int):
        raise TypeError(f"derivative order n must be an integer, it is {n}")
    if n < 0:
        raise ValueError(f"derivative order n must be non-negative, it is {n}")

    if n == 0:
        return u

    # Validate shape compatibility
    if u.ndim == 1:
        if d_cheb_matrix.shape[0] != d_cheb_matrix.shape[1]:
            raise ValueError(f"d_cheb_matrix must be square, got shape {d_cheb_matrix.shape}")
        if d_cheb_matrix.shape[0] != u.shape[0]:
            raise ValueError(
                f"Shape mismatch: d_cheb_matrix has shape {d_cheb_matrix.shape} "
                f"but u has length {u.shape[0]}. They must be compatible for "
                f"matrix-vector multiplication."
            )
    elif u.ndim == 2:
        if d_cheb_matrix.shape[0] != u.shape[0]:
            raise ValueError(
                f"Shape mismatch: d_cheb_matrix has shape {d_cheb_matrix.shape} "
                f"but u has shape {u.shape}. The first dimension must match."
            )
    else:
        raise ValueError(f"u must be 1D or 2D, got shape {u.shape}")

    # Compute derivative by matrix multiplication
    if n == 1:
        uxn = d_cheb_matrix @ u
    else:
        # Higher order derivatives: D^n @ u
        d_power = np.linalg.matrix_power(d_cheb_matrix, n)
        uxn = d_power @ u

    return uxn
