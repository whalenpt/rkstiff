r"""
Transform utilities for axisymmetric and spectral-space computations
====================================================================


This module implements the **Discrete Hankel Transform (DHT)** for
functions with cylindrical symmetry. The DHT provides a spectral
decomposition for radial domains analogous to the Fourier transform
for Cartesian coordinates, and is particularly useful in
axially symmetric PDEs, such as paraxial wave equations,
diffusion problems, or nonlinear Schrödinger-type systems.

Overview
--------

The Hankel transform pair of order zero is defined as:

.. math::

    G(k)
        &= \int_0^{\infty} f(r) J_0(k r)\, r \, dr, \\
    f(r)
        &= \int_0^{\infty} G(k) J_0(k r)\, k \, dk,

where :math:`J_0` is the Bessel function of the first kind of order zero.

The **Discrete Hankel Transform (DHT)** discretizes these integrals on
a finite interval :math:`r \in [0, r_{\max}]` using the zeros of
:math:`J_0`. This quasi-discrete formulation is spectrally accurate
for smooth, square-integrable functions on bounded domains.

Contents
--------

- :class:`HankelTransform` — Implements forward and inverse DHT.
"""

import numpy as np
import scipy.special as sp  # type: ignore
from .grids import construct_r_kr_hankel


class HankelTransform:
    r"""
    Discrete Hankel Transform (DHT) for axially symmetric functions.

    This class implements the quasi-discrete Hankel transform
    (QDHT) following Guizar-Sicairos and Gutiérrez-Vega (2004),
    suitable for efficiently transforming between real-space
    and spectral-space representations of radially symmetric functions.

    The continuous transform pair of order zero is:

    .. math::

        G(k) &= \int_0^\infty f(r) J_0(k r)\, r \, dr, \\
        f(r) &= \int_0^\infty G(k) J_0(k r)\, k \, dk,

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    The **discrete** version used here approximates the above integrals by
    evaluating :math:`f(r)` and :math:`G(k)` on the grids:

    .. math::

        r_i = \frac{j_i}{j_{N+1}}\,r_{\max}, \qquad
        k_i = \frac{j_i}{r_{\max}},

    where :math:`j_i` are the zeros of :math:`J_0`.

    Parameters
    ----------
    nr : int
        Number of radial sample points (≥ 4).
    rmax : float, optional
        Maximum radial domain size. Default is ``1.0``.

    Attributes
    ----------
    r : np.ndarray
        Radial grid :math:`[r_1, \ldots, r_N]`.
    kr : np.ndarray
        Spectral wavenumber grid corresponding to :math:`k_i`.
    _Y : np.ndarray
        Discrete Hankel transform matrix.
    _jN : float
        Normalization constant (the :math:`(N+1)`-th zero of :math:`J_0`).

    References
    ----------
    - Guizar-Sicairos, M. & Gutiérrez-Vega, J. C.
      *Computation of quasi-discrete Hankel transforms of integer order for propagating optical wave fields.*
      J. Opt. Soc. Am. A **21**, 53–58 (2004).
    """

    def __init__(self, nr: int, rmax: float = 1.0):
        """Initialize a discrete Hankel transform on a radial domain."""
        self._bessel_zeros = None
        self._jN = None
        self._Y = None
        self._setnr_setrmax(nr, rmax)

    def _setnr_setrmax(self, nr, rmax):
        """(Re)initialize radial and spectral grids when parameters change."""
        self._nr = nr
        self._rmax = rmax
        self.r, self.kr, self._bessel_zeros, self._jN = construct_r_kr_hankel(nr, rmax)
        self._sethankelmatrix()

    def _sethankelmatrix(self):
        """Construct the discrete Hankel transform matrix based on :math:`J_0` zeros."""
        j1vec = sp.j1(self._bessel_zeros)
        bessel_arg = np.outer(self._bessel_zeros, self._bessel_zeros) / self._jN
        self._Y = 2 * sp.j0(bessel_arg) / (self._jN * j1vec**2)

    def _scalefactor(self) -> float:
        r"""Internal scale factor :math:`S = r_{\max}^2 / j_{N+1}`."""
        return self.rmax**2 / self._jN

    # -------------------------------
    # User-facing API
    # -------------------------------

    def hankel_matrix(self) -> np.ndarray:
        r"""
        Return the discrete Hankel transform matrix.

        Returns
        -------
        np.ndarray, shape (nr, nr)
            Transform matrix :math:`Y_{ij} = 2 J_0(j_i j_j / j_{N+1}) / (j_{N+1} [J_1(j_j)]^2)`.
        """
        return self._Y.copy()

    def bessel_zeros(self) -> np.ndarray:
        r"""
        Return the zeros of the Bessel function :math:`J_0`.

        Returns
        -------
        np.ndarray
            Array of the first :math:`N` zeros of :math:`J_0`.
        """
        return self._bessel_zeros.copy()

    # -------------------------------
    # Properties
    # -------------------------------

    @property
    def nr(self) -> int:
        """Number of radial grid points."""
        return self._nr

    @nr.setter
    def nr(self, nr: int):
        """Set the number of radial grid points and reinitialize the transform."""
        if not isinstance(nr, int) or nr < 4:
            raise ValueError("nr must be an integer ≥ 4.")
        self._setnr_setrmax(nr, self._rmax)

    @property
    def rmax(self) -> float:
        """Maximum radius of the domain."""
        return self._rmax

    @rmax.setter
    def rmax(self, rmax: float):
        """Set maximum radial extent and recompute Hankel transform grid."""
        if rmax <= 0:
            raise ValueError("rmax must be positive.")
        self._setnr_setrmax(self._nr, rmax)

    # -------------------------------
    # Core transform routines
    # -------------------------------

    def ht(self, f: np.ndarray) -> np.ndarray:
        r"""
        Compute the **forward discrete Hankel transform**.

        Transforms a function :math:`f(r)` from physical (radial) space to
        spectral (wavenumber) space according to:

        .. math::

            G(k_i) \approx S \sum_{j=1}^{N}
                Y_{ij} f(r_j),

        where :math:`S = r_{\max}^2 / j_{N+1}` is the normalization factor.

        Parameters
        ----------
        f : np.ndarray
            Function values sampled at radial grid points ``r``.

        Returns
        -------
        np.ndarray
            Spectral coefficients sampled at corresponding wavenumbers ``kr``.

        Examples
        --------
        >>> ht = HankelTransform(nr=64, rmax=5.0)
        >>> f = np.exp(-ht.r**2)
        >>> G = ht.ht(f)
        """
        return self._scalefactor() * self._Y.dot(f)

    def iht(self, g: np.ndarray) -> np.ndarray:
        r"""
        Compute the **inverse discrete Hankel transform**.

        Transforms a spectral-space function :math:`G(k)` back to real-space
        :math:`f(r)` using:

        .. math::

            f(r_i) \approx \frac{1}{S}
            \sum_{j=1}^{N} Y_{ij} G(k_j),

        with the same transform matrix :math:`Y_{ij}`.

        Parameters
        ----------
        g : np.ndarray
            Spectral coefficients at wavenumber points ``kr``.

        Returns
        -------
        np.ndarray
            Real-space function evaluated at the radial grid ``r``.

        Examples
        --------
        >>> ht = HankelTransform(nr=64, rmax=5.0)
        >>> G = np.exp(-ht.kr**2)
        >>> f = ht.iht(G)
        """
        return self._Y.dot(g) / self._scalefactor()
