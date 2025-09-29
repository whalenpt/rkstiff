"""grids.py"""

from typing import Optional, Tuple
import numpy as np
import scipy.special as sp  # type: ignore


def construct_x_kx_rfft(n: int, a: float = 0.0, b: float = 2 * np.pi):
    """Constructs a uniform 1D spatial grid and rfft spectral wavenumbers for real-valued functions
    INPUTS
        n- even integer greater than 2
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - uniform 1D spatial grid
        kx - spectral wavenumber grid
    """

    if not isinstance(n, int):
        raise TypeError(f"Number of grid points nmust be an integer, it is {n}")
    if n <= 2:
        raise ValueError(f"Number of grid points n must be larger than 2, it is {n}")
    if (n % 2) != 0:
        raise ValueError("Integer n in construct_x_kx_rfft must be an even number")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    return x, kx


def construct_x_kx_fft(n: int, a: float = 0.0, b: float = 2 * np.pi):
    """Constructs a uniform 1D spatial grid and fft spectral wavenumbers for complex-valued functions
    INPUTS
        n - even integer greater than 2
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - uniform 1D spatial grid
        kx - spectral wavenumber grid
    """

    if not isinstance(n, int):
        raise TypeError(f"Number of grid points n must be an integer, it is {n}")
    if n <= 2:
        raise ValueError(f"Number of grid points n must be larger than 2, it is {n}")
    if (n % 2) != 0:
        raise ValueError("Integer n in construct_x_kx_rfft must be an even number")

    dx = (b - a) / n
    x = np.arange(a, b, dx)
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return x, kx


def construct_x_cheb(n: int, a: float = -1.0, b: float = 1.0):
    """Constructs a 1D grid with Chebyshev spatial discretization

    INPUTS
        n - positive integer
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - grid discretized at Chebyshev points
    """

    if not isinstance(n, int):
        raise TypeError(f"Max Chebyshev grid point number n must be an integer, it is {n}")
    if n < 2:
        raise ValueError(f"Max Chebyshev grid point number n must be larger than 1, it is {n}")
    x = np.polynomial.chebyshev.chebpts2(n + 1)
    x = a + (b - a) * (x + 1) / 2.0
    return x


def construct_x_dx_cheb(n: int, a: float = -1, b: float = 1):
    """
    Construct Chebyshev-Gauss-Lobatto grid points and the associated
    spectral differentiation matrix.

    Parameters
    ----------
    n : int
        Number of subintervals. The grid will contain N+1 Chebyshev points.
    a : float, optional
        Left endpoint of the interval (default: -1).
    b : float, optional
        Right endpoint of the interval (default: 1).

    Returns
    -------
    x : ndarray, shape (N+1,)
        Chebyshev-Gauss-Lobatto grid points mapped to [a, b].
    d_cheb_matrix : ndarray, shape (N+1, N+1)
        Differentiation matrix such that (d_cheb_matrix @ f) approximates df/dx
        for a function f sampled on the grid x.

    Notes
    -----
    The matrix is constructed using the barycentric interpolation
    formula (see Trefethen, *Spectral Methods in MATLAB*). It has the
    properties:
        - Rows sum to zero (consistent with differentiation).
        - Provides spectral accuracy for smooth functions.
    """
    x = construct_x_cheb(n, a, b)
    c = np.r_[2, np.ones(n - 1), 2] * np.power(-1, np.arange(0, n + 1))
    X = np.tile(x.reshape(n + 1, 1), (1, n + 1))  # put copies of x in columns (first row is x0)
    X = X - X.T
    d_cheb_matrix = np.outer(c, 1.0 / c) / (X + np.eye(n + 1))
    d_cheb_matrix = d_cheb_matrix - np.diag(d_cheb_matrix.sum(axis=1))
    return x, d_cheb_matrix


def construct_r_kr_hankel(nr: int, rmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct hankel transform radial and spectral grids
    Args:
        nr (int): number of radial points
        rmax (float): maximum radius of the radial grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            r - radial grid points
            kr - spectral grid points
            bessel_zeros - zeros of the Bessel function J0 used in constructing the grids
            jN - the Nth zero of the Bessel function J0 used in constructing the grids
    """
    # set the r and kr radial grids given a maximum radius rmax
    bessel_zeros = sp.jn_zeros(0, nr + 1)
    bessel_zeros, jN = bessel_zeros[:-1], bessel_zeros[-1]
    r = bessel_zeros * rmax / jN
    kr = bessel_zeros / rmax
    return r, kr, bessel_zeros, jN


class HankelTransform:
    """
    A class for computing discrete Hankel Transforms

    ATTRIBUTES
    __________

    nr : int
        Number of radial points sampled. The size of the
        hankel transform matrix is nr x nr.

    rmax : float
        Maximum radius of sampled points.

    r : np.array, dtype = float
        Radial points for a spectral grid suitable for the Hankel transform.
        This grid is determined by the user specification of nr and rmax

    kr : np.array, dtype = float
        Spectral points for a radial grid suitable for the Hankel transform.
        This grid is determined by the user specification of nr and rmax

    METHODS
    _______

    ht(f):
        compute_s a hankel transform of the function f sampled at the radial points
        specified by r.

    iht(g):
        compute_s an inverse hankel transform of the spectral space function g sampled
        at the spectral points specified by kr

    """

    def __init__(self, nr, rmax=1.0):
        """
        Constructs a Hankel transform Matrix that is used in the forward hankel transform
        function ht and inverse hankel transform function iht

        INPUTS
        ______

        nr : int
            Number of radial points sampled. The size of the
            hankel transform matrix is nr x nr. nr >= 4

        rmax :  float
            Maximum radius of sampled points. rmax > 0

        OUTPUTS
        _______
        None

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

    def hankel_matrix(self):
        """Returns a copy of the Hankel transform matrix constructed by the class."""
        return self._Y.copy()

    def bessel_zeros(self):
        """Returns a copy of the Bessel zeros used by the Hankel transform."""
        return self._bessel_zeros.copy()

    @property
    def nr(self):
        """Returns the number of radial points for the radial grid specified by the class."""
        return self._nr

    @nr.setter
    def nr(self, nr):
        """Sets the number of radial points in the grid to be used by the Hankel transform."""
        if not isinstance(nr, int):
            raise ValueError("nr must be an integer")
        if nr < 4:
            raise ValueError("nr must be greater than or equal to 4")
        self._nr = nr
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(nr, self._rmax)
        self._sethankelmatrix()

    @property
    def rmax(self):
        """Returns the maximum radius of the radial grid used by the Hankel transform"""
        return self._rmax

    @rmax.setter
    def rmax(self, rmax):
        """Sets the maximum radius of the radial grid used by the Hankel transform"""
        if rmax <= 0:
            raise ValueError("rmax must be non-negative")
        self._rmax = rmax
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(self.nr, self._rmax)

    def ht(self, f: np.ndarray) -> np.ndarray:
        """
        compute_s a hankel transform of the function f sampled at the radial
        points specified by r

        INPUTS
        ______

        f : np.array, dtype=float
            function sampled at the discretized points specified by r

        OUTPUTS
        _______

        g :  np.array, dtype=float
            Hankel spectral space representation of the function f corresponding
            to the spectral space grid points kr
        """

        return self._scalefactor() * self._Y.dot(f)

    def iht(self, g: np.ndarray) -> np.ndarray:
        """
        compute_s an inverse hankel transform of the function g sampled at the spectral
        space points specified by kr

        INPUTS
        ______

        g : np.array, dtype=float
            spectral space function sampled at the discretized points specified by kr

        OUTPUTS
        _______

        f :  np.array, dtype=float
            Real-space representation of the spectral space function g corresponding
            to values on the radial grid r
        """

        return self._Y.dot(g) / self._scalefactor()


def mirror_grid(r: np.ndarray, u: Optional[np.ndarray] = None, axis: int = -1):
    """
    Converts r grid from [0,rmax] interval to [-rmax,rmax] interval and adjusts
    function output u accordingly

    INPUTS
    ______

    r : np.array, dytpe=float
        radial grid on interval [0,rmax]

    u : np.array
        function values specified at radial points given by r

    axis : int
        axis value determines how to mirror the u array (-1 -> stack horizontally,
        0 -> stack vertically)

    OUTPUTS
    _______

    rnew : np.array, dtype=float
        'radial' grid on the interval [-rmax,rmax]

    unew : np.array
        function values specified at 'radial' points given by rnew

    """

    rnew = np.hstack([-np.flipud(r), r])
    if u is None:
        return rnew

    if axis == -1:
        unew = np.hstack([np.flipud(u), u])
    elif axis == 0:
        unew = np.vstack([np.flipud(u), u])
    elif axis == 1:
        unew = np.hstack([np.fliplr(u), u])
    else:
        raise ValueError("axis variable must be -1 or 0")

    return rnew, unew
