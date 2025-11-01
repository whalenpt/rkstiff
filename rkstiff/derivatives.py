"""rkstiff.derivatives

Provides functions for taking derivatives in spectral space using FFT methods
and Chebyshev spectral methods.
"""

import numpy as np


def dx_rfft(kx: np.ndarray, u: np.ndarray, n: int = 1) -> np.ndarray:
    r"""Compute the nth derivative of a real-valued array in spectral space.

    Uses the real FFT (rfft) to efficiently compute derivatives of real-valued
    functions. The derivative is computed by multiplying the Fourier coefficients
    by :math:`(ik_x)^n` and transforming back to real space.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid. For an input array of size N,
        kx should have size N//2 + 1 to match the rfft output size.
        Typically generated using ``np.fft.rfftfreq(N, d=dx)``.
    u : np.ndarray
        Real-valued input array. Must not be complex.
    n : int, optional
        Order of the derivative. Must be a non-negative integer.
        Default is 1 (first derivative).

    Returns
    -------
    np.ndarray
        The nth derivative of u, same shape as input u.

    Raises
    ------
    TypeError
        If n is not an integer or if u is complex-valued.
    ValueError
        If n is negative or if kx shape doesn't match rfft output shape.

    Notes
    -----
    For a real FFT, if the input array has size N, the rfft output has size
    N//2 + 1. Therefore, kx must have size N//2 + 1 to match the spectral
    coefficients.

    The derivative is computed as:

    .. math::
        \\frac{d^n u}{dx^n} = \\mathcal{F}^{-1}[(ik_x)^n \\mathcal{F}[u]]

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.sin(x)
    >>> dudx = dx_rfft(kx, u, n=1)  # First derivative
    >>> # dudx should be approximately cos(x)
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
    r"""Compute the nth derivative of a complex-valued array in spectral space.

    Uses the complex FFT (fft) to compute derivatives of complex-valued
    functions. The derivative is computed by multiplying the Fourier coefficients
    by :math:`(ik_x)^n` and transforming back to physical space.

    Parameters
    ----------
    kx : np.ndarray
        Wavenumbers of the spectral grid. Should have the same size as u.
        Typically generated using ``np.fft.fftfreq(N, d=dx)``.
    u : np.ndarray
        Complex-valued input array.
    n : int, optional
        Order of the derivative. Must be a non-negative integer.
        Default is 1 (first derivative).

    Returns
    -------
    np.ndarray
        The nth derivative of u, same shape as input u.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is negative or if kx shape doesn't match u shape.

    Notes
    -----
    For a complex FFT, kx should have the same size as the input array u.

    The derivative is computed as:

    .. math::
        \\frac{d^n u}{dx^n} = \\mathcal{F}^{-1}[(ik_x)^n \\mathcal{F}[u]]

    Examples
    --------
    >>> import numpy as np
    >>> N = 128
    >>> L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    >>> u = np.exp(1j * x)
    >>> dudx = dx_fft(kx, u, n=1)  # First derivative
    >>> # dudx should be approximately 1j * exp(1j * x)
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
    r"""Compute the nth derivative using Chebyshev spectral differentiation.

    Uses the Chebyshev differentiation matrix to compute derivatives of
    functions sampled at Chebyshev-Gauss-Lobatto points. Provides spectral
    accuracy for smooth functions.

    Parameters
    ----------
    d_cheb_matrix : np.ndarray, shape (N, N)
        Chebyshev differentiation matrix. Typically generated using
        ``construct_x_dx_cheb`` from ``rkstiff.grids``. The matrix should
        have shape (N, N) where N is the number of grid points.
    u : np.ndarray, shape (N,) or (N, M)
        Function values sampled at Chebyshev grid points. Can be 1D for a
        single function or 2D where each column represents a different function.
    n : int, optional
        Order of the derivative. Must be a non-negative integer.
        Default is 1 (first derivative).

    Returns
    -------
    np.ndarray
        The nth derivative of u, same shape as input u.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is negative or if d_cheb_matrix and u have incompatible shapes.

    Notes
    -----
    The derivative is computed by matrix multiplication:

    .. math::
        \\frac{du}{dx} = D u

    where :math:`D` is the Chebyshev differentiation matrix. Higher order
    derivatives are computed by repeated matrix multiplication:

    .. math::
        \\frac{d^n u}{dx^n} = D^n u

    This method is spectrally accurate (exponential convergence) for smooth
    functions but can suffer from numerical instability for very high order
    derivatives due to matrix power operations.

    For functions defined on an interval [a, b], the differentiation matrix
    must be properly scaled to account for the domain transformation.

    See Also
    --------
    rkstiff.grids.construct_x_dx_cheb : Construct Chebyshev grid and differentiation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.grids import construct_x_dx_cheb
    >>>
    >>> # Create Chebyshev grid and differentiation matrix
    >>> x, D = construct_x_dx_cheb(n=20, a=-1, b=1)
    >>>
    >>> # Test function: u = sin(x)
    >>> u = np.sin(x)
    >>>
    >>> # First derivative (should be cos(x))
    >>> dudx = dx_cheb(D, u, n=1)
    >>> dudx_exact = np.cos(x)
    >>> np.allclose(dudx, dudx_exact, atol=1e-10)
    True
    >>>
    >>> # Second derivative (should be -sin(x))
    >>> d2udx2 = dx_cheb(D, u, n=2)
    >>> d2udx2_exact = -np.sin(x)
    >>> np.allclose(d2udx2, d2udx2_exact, atol=1e-8)
    True
    >>>
    >>> # Multiple functions at once (2D array)
    >>> u_multi = np.column_stack([np.sin(x), np.cos(x)])
    >>> du_multi = dx_cheb(D, u_multi, n=1)
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
