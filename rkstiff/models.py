import numpy as np


def kdvSoliton(x, A=0.5, x0=0, t=0):
    u0 = 0.5 * A**2 / (np.cosh(A * (x - x0 - A**2 * t) / 2) ** 2)
    return u0


def kdvMultiSoliton(x, A, x0, t=0):
    assert len(x0) == len(A)
    M = len(A)
    N = len(x)
    A = np.array(A).reshape(1, M)
    x0 = np.array(x0).reshape(1, M)
    u0 = 0.5 * A**2 / (np.cosh(A * (x.reshape(N, 1) - x0 - A**2 * t) / 2) ** 2)
    u0 = np.sum(u0, axis=1)
    return u0


def kdvOps(kx):
    L = 1j * kx**3

    def NL(uf):
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -6 * np.fft.rfft(u * ux)

    return L, NL


def burgersOps(kx, mu):
    L = -mu * kx**2

    def NL(uf):
        u = np.fft.irfft(uf)
        ux = np.fft.irfft(1j * kx * uf)
        return -np.fft.rfft(u * ux)

    return L, NL


def allenCahnOps(x, Dx, epsilon):
    epsilon = 0.01
    D2 = Dx.dot(Dx)
    L = epsilon * D2 + np.eye(*D2.shape)
    L = L[1:-1, 1:-1]  # remove boundary points

    def NL(u):
        return x[1:-1] - np.power(u + x[1:-1], 3)

    return L, NL
