import numpy as np


def dx_rfft(kx, u, n=1):
    """Takes derivative(s) of a real valued array in spectral space
    INPUTS
        kx - wavenumbers of the spectral grid
        u - real valued array
        n - order of derivative (default is 1)
    OUTPUTS
        uxn - derivative of u to the nth power
    """
    if not isinstance(n, int):
        raise TypeError("derivative order n must be an integer, it is {}".format(n))
    if n < 0:
        raise ValueError("derivative order n must non-negative, it is {}".format(n))

    if n == 0:
        return u

    uFFT = np.fft.rfft(u)
    if n == 1:
        uxn = np.fft.irfft(1j * kx * uFFT)
    else:
        uxn = np.fft.irfft(np.power(1j * kx, n) * uFFT)
    return uxn


def dx_fft(kx, u, n=1):
    """Takes derivative(s) of a complex valued array in spectral space
    INPUTS
        kx - wavenumbers of the spectral grid
        u - complex valued array
        n - order of derivative (default is 1)
    OUTPUTS
        uxn - derivative of u to the nth power
    """

    if not isinstance(n, int):
        raise TypeError("derivative order n must be an integer, it is {}".format(n))
    if n < 0:
        raise ValueError("derivative order n must non-negative, it is {}".format(n))
    if n == 0:
        return u

    uFFT = np.fft.fft(u)
    if n == 1:
        uxn = np.fft.ifft(1j * kx * uFFT)
    else:
        uxn = np.fft.ifft(np.power(1j * kx, n) * uFFT)
    return uxn
