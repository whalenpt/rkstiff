"""grids.py

Grid construction utilities for spectral methods.

This module provides functions for constructing spatial and spectral grids
for various spectral methods including FFT-based methods, Chebyshev methods,
and Hankel transforms.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.special as sp  # type: ignore


def construct_x_kx_rfft(n: int, a: float = 0.0, b: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray]:
    """Construct uniform 1D spatial grid and rfft spectral wavenumbers.
    
    Creates a uniform spatial grid and corresponding wavenumber grid for
    real-valued functions using the real FFT (rfft) convention.
    
    Parameters
    ----------
    n : int
        Number of grid points. Must be an even integer greater than 2.
    a : float, optional
        Left endpoint of spatial grid. Default is 0.0.
    b : float, optional
        Right endpoint of spatial grid. Default is :math:`2\\pi`.
    
    Returns
    -------
    x : np.ndarray
        Uniform 1D spatial grid with n points in the interval [a, b).
    kx : np.ndarray
        Spectral wavenumber grid with n//2 + 1 points for rfft.
    
    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n <= 2 or if n is not even.
    
    Examples
    --------
    >>> x, kx = construct_x_kx_rfft(128, a=0, b=2*np.pi)
    >>> x.shape
    (128,)
    >>> kx.shape
    (65,)
    
    Notes
    -----
    The grid spacing is :math:`dx = (b-a)/n` and wavenumbers are computed
    as :math:`k_x = 2\\pi \\cdot \\text{rfftfreq}(n, dx)`.
    """
    if not isinstance(n, int):
        raise TypeError(f"Number of grid points n must be an integer, it is {n}")
    if n <= 2:
        raise ValueError(f"Number of grid points n must be larger than 2, it is {n}")
    if (n % 2) != 0:
        raise ValueError("Integer n in construct_x_kx_rfft must be an even number")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    return x, kx


def construct_x_kx_fft(n: int, a: float = 0.0, b: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray]:
    """Construct uniform 1D spatial grid and fft spectral wavenumbers.
    
    Creates a uniform spatial grid and corresponding wavenumber grid for
    complex-valued functions using the complex FFT (fft) convention.
    
    Parameters
    ----------
    n : int
        Number of grid points. Must be an even integer greater than 2.
    a : float, optional
        Left endpoint of spatial grid. Default is 0.0.
    b : float, optional
        Right endpoint of spatial grid. Default is :math:`2\\pi`.
    
    Returns
    -------
    x : np.ndarray
        Uniform 1D spatial grid with n points in the interval [a, b).
    kx : np.ndarray
        Spectral wavenumber grid with n points for fft.
    
    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n <= 2 or if n is not even.
    
    Examples
    --------
    >>> x, kx = construct_x_kx_fft(128, a=0, b=2*np.pi)
    >>> x.shape
    (128,)
    >>> kx.shape
    (128,)
    
    Notes
    -----
    The grid spacing is :math:`dx = (b-a)/n` and wavenumbers are computed
    as :math:`k_x = 2\\pi \\cdot \\text{fftfreq}(n, dx)`.
    """
    if not isinstance(n, int):
        raise TypeError(f"Number of grid points n must be an integer, it is {n}")
    if n <= 2:
        raise ValueError(f"Number of grid points n must be larger than 2, it is {n}")
    if (n % 2) != 0:
        raise ValueError("Integer n in construct_x_kx_fft must be an even number")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return x, kx


def construct_x_cheb(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Construct 1D grid with Chebyshev spatial discretization.
    
    Creates a grid using Chebyshev-Gauss-Lobatto points (also known as
    Chebyshev extreme points), which are optimal for spectral methods
    and polynomial interpolation.
    
    Parameters
    ----------
    n : int
        Maximum Chebyshev grid point index. The grid will contain n+1 points.
        Must be at least 2.
    a : float, optional
        Left endpoint of spatial grid. Default is -1.0.
    b : float, optional
        Right endpoint of spatial grid. Default is 1.0.
    
    Returns
    -------
    np.ndarray
        Grid of n+1 points discretized at Chebyshev points in the interval [a, b].
    
    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n < 2.
    
    Notes
    -----
    The Chebyshev points on [-1, 1] are given by:
    
    .. math::
        x_j = \\cos\\left(\\frac{j\\pi}{n}\\right), \\quad j = 0, 1, \\ldots, n
    
    These points are then linearly mapped to the interval [a, b].
    
    Examples
    --------
    >>> x = construct_x_cheb(10, a=-1, b=1)
    >>> len(x)
    11
    >>> x[0], x[-1]  # endpoints
    (1.0, -1.0)
    """
    if not isinstance(n, int):
        raise TypeError(f"Max Chebyshev grid point number n must be an integer, it is {n}")
    if n < 2:
        raise ValueError(f"Max Chebyshev grid point number n must be larger than 1, it is {n}")
    x = np.polynomial.chebyshev.chebpts2(n + 1)
    x = a + (b - a) * (x + 1) / 2.0
    return x


def construct_x_dx_cheb(n: int, a: float = -1, b: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Chebyshev-Gauss-Lobatto grid and differentiation matrix.
    
    Creates both the Chebyshev grid points and the associated spectral
    differentiation matrix for computing derivatives.
    
    Parameters
    ----------
    n : int
        Number of subintervals. The grid will contain n+1 Chebyshev points.
        Must be at least 2.
    a : float, optional
        Left endpoint of the interval. Default is -1.
    b : float, optional
        Right endpoint of the interval. Default is 1.
    
    Returns
    -------
    x : np.ndarray, shape (n+1,)
        Chebyshev-Gauss-Lobatto grid points mapped to [a, b].
    d_cheb_matrix : np.ndarray, shape (n+1, n+1)
        Differentiation matrix such that ``d_cheb_matrix @ f`` approximates
        df/dx for a function f sampled on the grid x.
    
    Notes
    -----
    The matrix is constructed using the barycentric interpolation
    formula [1]_. It has the following properties:
    
    - Rows sum to zero (consistent with differentiation of constants)
    - Provides spectral accuracy for smooth functions
    - Eigenvalues lie within the unit circle for stability
    
    The differentiation matrix is given by:
    
    .. math::
        D_{ij} = \\begin{cases}
        \\frac{c_i}{c_j} \\frac{1}{x_i - x_j} & i \\neq j \\\\
        -\\sum_{k \\neq i} D_{ik} & i = j
        \\end{cases}
    
    where :math:`c_i` are barycentric weights.
    
    References
    ----------
    .. [1] Trefethen, Lloyd N. "Spectral methods in MATLAB." Society for
           industrial and applied mathematics, 2000.
    
    Examples
    --------
    >>> x, D = construct_x_dx_cheb(10, a=-1, b=1)
    >>> f = np.sin(x)
    >>> dfdx = D @ f  # Approximate derivative
    >>> dfdx_exact = np.cos(x)
    >>> np.allclose(dfdx, dfdx_exact, atol=1e-10)
    True
    """
    x = construct_x_cheb(n, a, b)
    c = np.r_[2, np.ones(n - 1), 2] * np.power(-1, np.arange(0, n + 1))
    X = np.tile(x.reshape(n + 1, 1), (1, n + 1))  # put copies of x in columns (first row is x0)
    X = X - X.T
    d_cheb_matrix = np.outer(c, 1.0 / c) / (X + np.eye(n + 1))
    d_cheb_matrix = d_cheb_matrix - np.diag(d_cheb_matrix.sum(axis=1))
    return x, d_cheb_matrix


def construct_r_kr_hankel(nr: int, rmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct Hankel transform radial and spectral grids.
    
    Creates optimal grids for discrete Hankel transforms based on zeros
    of the Bessel function :math:`J_0`.
    
    Parameters
    ----------
    nr : int
        Number of radial points. Must be at least 1.
    rmax : float
        Maximum radius of the radial grid. Must be positive.
    
    Returns
    -------
    r : np.ndarray
        Radial grid points in the interval (0, rmax].
    kr : np.ndarray
        Spectral grid points in wavenumber space.
    bessel_zeros : np.ndarray
        First nr zeros of the Bessel function :math:`J_0`.
    jN : float
        The (nr+1)-th zero of :math:`J_0`, used for normalization.
    
    Notes
    -----
    The grids are constructed using zeros of :math:`J_0(x)` as:
    
    .. math::
        r_j &= \\frac{\\alpha_j r_{\\text{max}}}{\\alpha_{n+1}} \\\\
        k_j &= \\frac{\\alpha_j}{r_{\\text{max}}}
    
    where :math:`\\alpha_j` are the zeros of :math:`J_0`.
    
    Examples
    --------
    >>> r, kr, zeros, jN = construct_r_kr_hankel(10, rmax=5.0)
    >>> len(r), len(kr)
    (10, 10)
    >>> r[-1] < 5.0  # Last point is less than rmax
    True
    """
    # set the r and kr radial grids given a maximum radius rmax
    if nr < 4:
        raise ValueError("nr must be greater than or equal to 4")
    if rmax <= 0:
        raise ValueError("rmax must be positive") 
    bessel_zeros = sp.jn_zeros(0, nr + 1)
    bessel_zeros, jN = bessel_zeros[:-1], bessel_zeros[-1]
    r = bessel_zeros * rmax / jN
    kr = bessel_zeros / rmax
    return r, kr, bessel_zeros, jN


class HankelTransform:
    """Discrete Hankel Transform for axially symmetric functions.
    
    Implements the discrete Hankel transform (DHT) and its inverse for
    functions with cylindrical symmetry. The transform is based on the
    quasi-discrete Hankel transform using zeros of Bessel functions.
    
    Parameters
    ----------
    nr : int
        Number of radial points sampled. The size of the Hankel transform
        matrix is nr x nr. Must be at least 4.
    rmax : float, optional
        Maximum radius of sampled points. Must be positive. Default is 1.0.
    
    Attributes
    ----------
    r : np.ndarray
        Radial points for the spectral grid suitable for the Hankel transform.
    kr : np.ndarray
        Spectral points in wavenumber space.
    nr : int
        Number of radial points.
    rmax : float
        Maximum radius of the radial grid.
    
    Methods
    -------
    ht(f)
        Compute forward Hankel transform.
    iht(g)
        Compute inverse Hankel transform.
    hankel_matrix()
        Return the Hankel transform matrix.
    bessel_zeros()
        Return the Bessel function zeros used.
    
    Notes
    -----
    The Hankel transform pair is defined as:
    
    .. math::
        G(k) &= \\int_0^\\infty f(r) J_0(kr) r \\, dr \\\\
        f(r) &= \\int_0^\\infty G(k) J_0(kr) k \\, dk
    
    where :math:`J_0` is the Bessel function of the first kind of order 0.
    
    References
    ----------
    .. [1] Guizar-Sicairos, M., & Gutierrez-Vega, J. C. (2004). Computation of
           quasi-discrete Hankel transforms of integer order for propagating
           optical wave fields. JOSA A, 21(1), 53-58.
    
    Examples
    --------
    >>> ht = HankelTransform(nr=64, rmax=10.0)
    >>> r = ht.r
    >>> f = np.exp(-r**2)  # Gaussian in real space
    >>> F = ht.ht(f)  # Transform to spectral space
    >>> f_reconstructed = ht.iht(F)  # Transform back
    >>> np.allclose(f, f_reconstructed)
    True
    """

    def __init__(self, nr: int, rmax: float = 1.0):
        """Initialize HankelTransform with specified grid parameters.
        
        Parameters
        ----------
        nr : int
            Number of radial points sampled. Must be at least 4.
        rmax : float, optional
            Maximum radius of sampled points. Must be positive. Default is 1.0.
        
        Raises
        ------
        ValueError
            If nr < 4 or if rmax <= 0.
        """
        self._bessel_zeros = None
        self._jN = None
        self._Y = None
        self._setnr_setrmax(nr, rmax)

    def _setnr_setrmax(self, nr, rmax):
        self._nr = nr
        self._rmax = rmax
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(nr, rmax)
        self._sethankelmatrix()

    def _sethankelmatrix(self):
        # set the Hankel matrix used in the Hankel transform for the saved radial grid
        j1vec = sp.j1(self._bessel_zeros)  # Bessel J1  # pylint: disable=E1101
        bessel_arg = np.outer(self._bessel_zeros, self._bessel_zeros) / self._jN
        self._Y = 2 * sp.j0(bessel_arg) / (self._jN * j1vec**2)  # Bessel J0  # pylint: disable=E1101

    def _scalefactor(self):
        # factor used in transforming from real space to spectral space and vice_versa for internal use
        return self.rmax**2 / self._jN

    def hankel_matrix(self) -> np.ndarray:
        """Return a copy of the Hankel transform matrix.
        
        Returns
        -------
        np.ndarray, shape (nr, nr)
            The Hankel transform matrix used for forward and inverse transforms.
        """
        return self._Y.copy()

    def bessel_zeros(self) -> np.ndarray:
        """Return a copy of the Bessel zeros used by the transform.
        
        Returns
        -------
        np.ndarray
            The first nr zeros of the Bessel function :math:`J_0`.
        """
        return self._bessel_zeros.copy()

    @property
    def nr(self) -> int:
        """Number of radial points in the grid."""
        return self._nr

    @nr.setter
    def nr(self, nr: int):
        """Set the number of radial points.
        
        Parameters
        ----------
        nr : int
            Number of radial points. Must be at least 4.
        
        Raises
        ------
        ValueError
            If nr is not an integer or if nr < 4.
        """
        if not isinstance(nr, int):
            raise ValueError("nr must be an integer")
        if nr < 4:
            raise ValueError("nr must be greater than or equal to 4")
        self._nr = nr
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(nr, self._rmax)
        self._sethankelmatrix()

    @property
    def rmax(self) -> float:
        """Maximum radius of the radial grid."""
        return self._rmax

    @rmax.setter
    def rmax(self, rmax: float):
        """Set the maximum radius of the radial grid.
        
        Parameters
        ----------
        rmax : float
            Maximum radius. Must be positive.
        
        Raises
        ------
        ValueError
            If rmax <= 0.
        """
        if rmax <= 0:
            raise ValueError("rmax must be positive")
        self._rmax = rmax
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(self.nr, self._rmax)

    def ht(self, f: np.ndarray) -> np.ndarray:
        """Compute forward Hankel transform.
        
        Transforms a function from real space to spectral space.
        
        Parameters
        ----------
        f : np.ndarray
            Function values sampled at the radial grid points r.
        
        Returns
        -------
        np.ndarray
            Hankel spectral space representation corresponding to
            the spectral grid points kr.
        
        Examples
        --------
        >>> ht = HankelTransform(nr=32, rmax=5.0)
        >>> f = np.exp(-ht.r**2)
        >>> F = ht.ht(f)
        """
        return self._scalefactor() * self._Y.dot(f)

    def iht(self, g: np.ndarray) -> np.ndarray:
        """Compute inverse Hankel transform.
        
        Transforms a function from spectral space back to real space.
        
        Parameters
        ----------
        g : np.ndarray
            Spectral space function sampled at the spectral grid points kr.
        
        Returns
        -------
        np.ndarray
            Real-space representation corresponding to values on the
            radial grid r.
        
        Examples
        --------
        >>> ht = HankelTransform(nr=32, rmax=5.0)
        >>> F = np.exp(-ht.kr**2)
        >>> f = ht.iht(F)
        """
        return self._Y.dot(g) / self._scalefactor()


def mirror_grid(
    r: np.ndarray, 
    u: Optional[np.ndarray] = None, 
    axis: int = -1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Mirror a radial grid to create a symmetric interval.
    
    Converts a radial grid from [0, rmax] to [-rmax, rmax] by mirroring
    the points and (optionally) the corresponding function values.
    
    Parameters
    ----------
    r : np.ndarray
        Radial grid on interval [0, rmax].
    u : np.ndarray, optional
        Function values at radial points r. If None, only the mirrored
        grid is returned.
    axis : int, optional
        Axis along which to mirror the u array. Default is -1.
        
        * -1 : Stack horizontally (for 1D arrays)
        * 0 : Stack vertically (rows)
        * 1 : Stack horizontally (columns)
    
    Returns
    -------
    rnew : np.ndarray
        Mirrored 'radial' grid on interval [-rmax, rmax].
    unew : np.ndarray, optional
        Function values at mirrored grid points. Only returned if u is provided.
    
    Raises
    ------
    ValueError
        If axis is not -1, 0, or 1.
    
    Examples
    --------
    >>> r = np.array([0, 1, 2, 3])
    >>> u = np.array([1, 2, 3, 4])
    >>> rnew, unew = mirror_grid(r, u)
    >>> rnew
    array([-3, -2, -1,  0,  0,  1,  2,  3])
    >>> unew
    array([4, 3, 2, 1, 1, 2, 3, 4])
    
    Notes
    -----
    This is useful for visualizing radially symmetric solutions or for
    problems that require symmetric boundary conditions.
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
        raise ValueError("axis variable must be -1, 0, or 1")

    return rnew, unew
