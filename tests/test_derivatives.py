""" Tests for the derivative functions."""
import numpy as np
from rkstiff.grids import construct_x_kx_rfft, construct_x_kx_fft
from rkstiff.grids import construct_x_dx_cheb
from rkstiff.derivatives import dx_rfft, dx_fft


def test_periodic_dx_rfft():
    """ Test the Fourier derivative on a periodic function."""
    n = 100
    a, b = 0, 2 * np.pi
    x, kx = construct_x_kx_rfft(n, a, b)
    u = np.sin(x)
    ux_exact = np.cos(x)
    ux_approx = dx_rfft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_zeroboundaries_dx_rfft():
    """ Test the Fourier derivative on a function with zero boundaries."""
    n = 400
    a, b = -30.0, 30.0
    x, kx = construct_x_kx_rfft(n, a, b)
    u = 1.0 / np.cosh(x)
    ux_exact = -np.tanh(x) / np.cosh(x)
    ux_approx = dx_rfft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_gauss_dx_rfft():
    """ Test the Fourier derivative on a Gaussian function."""
    n = 128
    a, b = -10, 10
    x, kx = construct_x_kx_rfft(n, a, b)
    u = np.exp(-(x**2))
    ux_exact = -2 * x * np.exp(-(x**2))
    ux_approx = dx_rfft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_manydx_rfft():
    """ Test multiple applications of the Fourier derivative."""
    n = 128
    a, b = 0, 2 * np.pi
    x, kx = construct_x_kx_rfft(n, a, b)
    u = np.sin(x)
    ux_exact = np.sin(x)

    ux_approx = u.copy()
    for _ in range(4):
        ux_approx = dx_rfft(kx, ux_approx)
    rel_err = np.linalg.norm(ux_exact - ux_approx) / np.linalg.norm(ux_exact)
    assert rel_err < 1e-6

    ux_approx = u.copy()
    ux_approx = dx_rfft(kx, ux_approx, 8)
    rel_err = np.linalg.norm(ux_exact - ux_approx) / np.linalg.norm(ux_exact)
    assert rel_err < 0.1


def test_manydx_fft():
    """ Test multiple applications of the Fourier derivative."""
    n = 128
    a, b = 0, 2 * np.pi
    x, kx = construct_x_kx_fft(n, a, b)
    u = np.sin(x)
    ux_exact = np.sin(x)

    ux_approx = u.copy()
    for _ in range(4):
        ux_approx = dx_fft(kx, ux_approx)
    rel_err = np.linalg.norm(ux_exact - ux_approx) / np.linalg.norm(ux_exact)
    assert rel_err < 1e-6

    ux_approx = u.copy()
    ux_approx = dx_fft(kx, ux_approx, 8)
    rel_err = np.linalg.norm(ux_exact - ux_approx) / np.linalg.norm(ux_exact)
    assert rel_err < 0.1


def test_periodic_dx_fft():
    """ Test the Fourier derivative on a periodic function."""
    n = 100
    a, b = 0, 2 * np.pi
    x, kx = construct_x_kx_fft(n, a, b)
    u = np.sin(x)
    ux_exact = np.cos(x)
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_zeroboundaries_dx_fft():
    """ Test the Fourier derivative on a function with zero boundaries."""
    n = 400
    a, b = -30.0, 30.0
    x, kx = construct_x_kx_fft(n, a, b)
    u = 1.0 / np.cosh(x)
    ux_exact = -np.tanh(x) / np.cosh(x)
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_gauss_dx_fft():
    """ Test the Fourier derivative on a Gaussian function."""
    n = 128
    a, b = -10, 10
    x, kx = construct_x_kx_fft(n, a, b)
    u = np.exp(-(x**2))
    ux_exact = -2 * x * np.exp(-(x**2))
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_exp_trig_dx_cheb():
    """ Test the Chebyshev derivative on a non-periodic function."""
    # standard interval [-1,1]
    n = 20
    a = -1
    b = 1
    x, d_cheb_matrix = construct_x_dx_cheb(n, -1, 1)
    u = np.exp(x) * np.sin(5 * x)
    du_exact = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    du_approx = d_cheb_matrix.dot(u)
    error = du_exact - du_approx
    assert np.linalg.norm(error) / np.linalg.norm(du_exact) < 1e-8

    # non-standard interval [-3,3]
    n = 30
    a = -3
    b = 3
    x, d_cheb_matrix = construct_x_dx_cheb(n, a, b)
    u = np.exp(x) * np.sin(5 * x)
    du_exact = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    du_approx = d_cheb_matrix.dot(u)
    error = du_exact - du_approx
    assert np.linalg.norm(error) / np.linalg.norm(du_exact) < 1e-7
