r"""
Spectral differentiation utilities for Fourier and Chebyshev grids
==================================================================

This module provides high-accuracy derivative operators using
spectral methods suitable for **stiff PDE solvers**.
Each function supports NumPy arrays and implements differentiation
via spectral transforms or Chebyshev differentiation matrices.

Overview
--------

Spectral differentiation is based on the principle that differentiation
in physical space corresponds to multiplication by :math:`(i k_x)^n`
in Fourier space, or repeated application of a differentiation matrix
in Chebyshev space:

.. math::

    \frac{d^n u}{dx^n}
        = \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right],
        \quad
        \text{(Fourier methods)}

and

.. math::

    \frac{d^n u}{dx^n} = D^n u,
        \quad
        \text{(Chebyshev methods)},

where :math:`\mathcal{F}` denotes the Fourier transform
and :math:`D` is the Chebyshev differentiation matrix.

Functions
---------

- :func:`dx_rfft` — Derivatives using the *real* FFT (rFFT)
- :func:`dx_fft` — Derivatives using the *complex* FFT
- :func:`dx_cheb` — Derivatives using the Chebyshev differentiation matrix
"""

import numpy as np


def dx_rfft(kx: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative of a **real-valued** array using the real FFT (rFFT).

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid.
        For an input size :math:`N`, ``kx`` must have size :math:`N/2 + 1`.
        Typically generated via ``np.fft.rfftfreq(N, d=dx) * 2π``.
    u : np.ndarray
        Real-valued input array of length :math:`N`. Must **not** be complex.
    n : int, optional
        Derivative order (non-negative integer). Default is ``1``.

    Returns
    -------
    np.ndarray
        The *n*-th derivative of ``u`` (real-valued array of same shape).

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``u`` is complex-valued.
    ValueError
        If ``n`` is negative or ``kx`` has incompatible shape.

    Notes
    -----
    The derivative is computed using the spectral relation:

    .. math::

        \frac{d^n u}{dx^n}
            = \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right],

    where :math:`\mathcal{F}` is the discrete Fourier transform (rFFT variant).

    For a real-valued input array of size :math:`N`, the rFFT output has size
    :math:`N/2 + 1`. Therefore, ``kx`` must match that shape exactly.

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.sin(x)
    >>> dudx = dx_rfft(kx, u)
    >>> np.allclose(dudx, np.cos(x), atol=1e-10)
    True
    """
    if not isinstance(n, int):
        raise TypeError(f"derivative order n must be an integer, it is {n}")
    if n < 0:
        raise ValueError(f"derivative order n must be non-negative, it is {n}")

    if np.iscomplexobj(u):
        raise TypeError("dx_rfft requires real-valued input. Use dx_fft for complex arrays.")

    if u.size == 0:
        return np.array([], dtype=float)

    if n == 0:
        return u

    u_fft = np.fft.rfft(u)

    if kx.shape != u_fft.shape:
        raise ValueError(
            f"kx shape {kx.shape} does not match rFFT output shape {u_fft.shape}. "
            "For input size N, kx should have size N//2 + 1."
        )

    uxn = np.fft.irfft((1j * kx) ** n * u_fft, n=u.shape[-1])
    return uxn


def dx_fft(kx: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative of a **complex-valued** array using the FFT.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid. Must match ``u.shape``.
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
    Fourier differentiation rule:

    .. math::

        \frac{d^n u}{dx^n}
            = \mathcal{F}^{-1}\!\left[(i k_x)^n \mathcal{F}[u]\right].

    This method is applicable to periodic complex fields or spectral-space data.

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.exp(1j * x)
    >>> dudx = dx_fft(kx, u)
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
    if kx.shape != u_fft.shape:
        raise ValueError(f"kx shape {kx.shape} must match FFT output {u_fft.shape}")

    return np.fft.ifft((1j * kx) ** n * u_fft)


def dx_cheb(d_cheb_matrix: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""
    Compute the *n*-th derivative using a **Chebyshev spectral differentiation matrix**.

    Parameters
    ----------
    d_cheb_matrix : np.ndarray
        Chebyshev differentiation matrix :math:`D \in \mathbb{R}^{N\times N}`,
        typically constructed with ``construct_x_dx_cheb``.
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
    The Chebyshev differentiation operator is applied as:

    .. math::

        \frac{du}{dx} = D u, \qquad
        \frac{d^n u}{dx^n} = D^n u.

    For functions defined on a domain :math:`[a,b]`, the differentiation matrix
    must be scaled by :math:`2/(b-a)` to account for mapping from :math:`[-1,1]`.

    This method achieves **spectral accuracy** for smooth functions,
    but high derivative orders may amplify rounding errors.

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.grids import construct_x_dx_cheb
    >>> x, D = construct_x_dx_cheb(n=20, a=-1, b=1)
    >>> u = np.sin(x)
    >>> dudx = dx_cheb(D, u)
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

    if u.ndim == 1:
        if d_cheb_matrix.shape[0] != d_cheb_matrix.shape[1]:
            raise ValueError("d_cheb_matrix must be square.")
        if d_cheb_matrix.shape[0] != u.shape[0]:
            raise ValueError("Shape mismatch: d_cheb_matrix and u must align in the first dimension.")
    elif u.ndim == 2:
        if d_cheb_matrix.shape[0] != u.shape[0]:
            raise ValueError("Shape mismatch: d_cheb_matrix and u must align in the first dimension.")
    else:
        raise ValueError(f"u must be 1D or 2D, got shape {u.shape}")

    if n == 1:
        return d_cheb_matrix @ u

    d_power = np.linalg.matrix_power(d_cheb_matrix, n)
    return d_power @ u
