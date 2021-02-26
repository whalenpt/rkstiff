
import numpy as np
import scipy.special as sp

def construct_x_kx_rfft(N,a=0.0,b=2*np.pi):
    """ Constructs a uniform 1D spatial grid and rfft spectral wavenumbers for real-valued functions
    INPUTS
        N - even integer greater than 2
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - uniform 1D spatial grid
        kx - spectral wavenumber grid
    """

    if not isinstance(N,int):
        raise TypeError('Number of grid points N must be an integer, it is {}'.format(N))
    if N <= 2:
        raise ValueError('Number of grid points N must be larger than 2, it is {}'.format(N))
    if (N % 2) != 0:
        raise ValueError('Integer N in construct_x_kx_rfft must be an even number')

    dx = (b-a)/N
    x = np.arange(a,b,dx)
    kx = 2*np.pi*np.fft.rfftfreq(N,d=dx)
    return x,kx


def construct_x_kx_fft(N,a=0.0,b=2*np.pi):
    """ Constructs a uniform 1D spatial grid and fft spectral wavenumbers for complex-valued functions
    INPUTS
        N - even integer greater than 2
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - uniform 1D spatial grid
        kx - spectral wavenumber grid
    """

    if not isinstance(N,int):
        raise TypeError('Number of grid points N must be an integer, it is {}'.format(N))
    if N <= 2:
        raise ValueError('Number of grid points N must be larger than 2, it is {}'.format(N))
    if (N % 2) != 0:
        raise ValueError('Integer N in construct_x_kx_rfft must be an even number')

    dx = (b-a)/N
    x = np.arange(a,b,dx)
    kx = 2*np.pi*np.fft.fftfreq(N,d=dx)
    return x,kx

def construct_x_cheb(N,a=-1,b=1):
    """ Constructs a 1D grid with Chebyshev spatial discretization 

    INPUTS
        N - positive integer
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - grid discretized at Chebyshev points
    """

    if not isinstance(N,int):
        raise TypeError('Max Chebyshev grid point number N must be an integer, it is {}'.format(N))
    if N < 2:
        raise ValueError('Max Chebyshev grid point number N must be larger than 1, it is {}'.format(N))
    x = np.polynomial.chebyshev.chebpts2(N+1)
    x = a+(b-a)*(x+1)/2.
    return x

def construct_x_Dx_cheb(N,a=-1,b=1):
    """ Constructs a 1D grid with Chebyshev spatial discretization along with a
        differentiation matrix for functions sampled on this grid

    INPUTS
        N - positive integer
        a - left endpoint in spatial grid
        b - right endpoint in spatial grid
    OUTPUTS
        x - grid discretized at Chebyshev points
        Dx - derivative matrix for functions sampled at Chebyshev points
    """
    x = construct_x_cheb(N,a,b)
    c = np.r_[2,np.ones(N-1),2]*np.power(-1,np.arange(0,N+1))
    X = np.tile(x.reshape(N+1,1),(1,N+1)) # put copies of x in columns (first row is x0)
    dX = X - X.T
    Dx = np.outer(c,1./c)/(dX+np.eye(N+1))
    Dx = Dx - np.diag(Dx.sum(axis=1))
    return x,Dx


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
        Computes a hankel transform of the function f sampled at the radial points 
        specified by r.

    iht(g):
        Computes an inverse hankel transform of the spectral space function g sampled
        at the spectral points specified by kr

    """

    def __init__(self,nr,rmax=1.0):
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
        self._setnr_setrmax(nr,rmax)

    def _setnr_setrmax(self,nr,rmax):
        self._setbesselzeros(nr)
        self._setgrids(rmax)
        self._sethankelmatrix()
        self._rmax = rmax
        self._nr = nr 

    def _setbesselzeros(self,nr):
        # find and save first nr bessel zeros in an array, also save the nr+1 bessel zero
        bessel_zeros = sp.jn_zeros(0,nr+1)
        self._bessel_zeros,self._jN = bessel_zeros[:-1], bessel_zeros[-1]

    def _setgrids(self,rmax):
        # set the r and kr radial grids given a maximum radius rmax
        self.r = self._bessel_zeros*rmax/self._jN
        self.kr = self._bessel_zeros/rmax

    def _sethankelmatrix(self):
        # set the Hankel matrix used in the Hankel transform for the saved radial grid
        j1vec = sp.j1(self._bessel_zeros)
        bessel_arg = np.outer(self._bessel_zeros,self._bessel_zeros)/self._jN
        self._Y = 2*sp.j0(bessel_arg)/(self._jN*j1vec**2)

    def _scalefactor(self):
        # factor used in transforming from real space to spectral space and vice_versa for internal use
        return self.rmax**2/self._jN

    def hankelMatrix(self):
        """ Returns a copy of the Hankel transform matrix constructed by the class. """
        return self._Y.copy()

    def besselZeros(self):
        """ Returns a copy of the Bessel zeros used by the Hankel transform. """
        return self._bessel_zeros.copy()

    @property
    def nr(self):
        """ Returns the number of radial points for the radial grid specified by the class. """
        return self._nr

    @nr.setter
    def nr(self,nr):
        """ Sets the number of radial points in the grid to be used by the Hankel transform. """
        if not isinstance(nr,int):
            raise ValueError('nr must be an integer')
        if nr < 4: 
            raise ValueError('nr must be greater than or equal to 4')
        self._setbesselzeros(nr)
        self._setgrids(self.rmax)
        self._sethankelmatrix()
        self._nr = nr 

    @property
    def rmax(self):
        """ Returns the maximum radius of the radial grid used by the Hankel transform """
        return self._rmax

    @rmax.setter
    def rmax(self,rmax):
        """ Sets the maximum radius of the radial grid used by the Hankel transform """
        if rmax <= 0:
            raise ValueError('rmax must be non-negative')
        self._setgrids(rmax)
        self._rmax = rmax 

    def ht(self,f):
        """
        Computes a hankel transform of the function f sampled at the radial
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

        return self._scalefactor()*self._Y.dot(f)

    def iht(self,g):
        """
        Computes an inverse hankel transform of the function g sampled at the spectral
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

        return self._Y.dot(g)/self._scalefactor()


def mirrorGrid(r,u=None,axis=-1):
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

    rnew = np.hstack([-np.flipud(r),r])
    if u is None:
        return rnew

    if axis == -1:
        unew = np.hstack([np.flipud(u),u])
    elif axis == 0:
        unew = np.vstack([np.flipud(u),u])
    elif axis == 1:
        unew = np.hstack([np.fliplr(u),u])
    else:
        raise ValueError('axis variable must be -1 or 0')

    return rnew,unew


