"""Tests for the derivative functions."""

import pytest
import numpy as np
from rkstiff.grids import construct_x_kx_fft
from rkstiff.derivatives import dx_fft


def test_manydx_fft():
    """Test multiple applications of the Fourier derivative."""
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
    """Test the Fourier derivative on a periodic function."""
    n = 100
    a, b = 0, 2 * np.pi
    x, kx = construct_x_kx_fft(n, a, b)
    u = np.sin(x)
    ux_exact = np.cos(x)
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_zeroboundaries_dx_fft():
    """Test the Fourier derivative on a function with zero boundaries."""
    n = 400
    a, b = -30.0, 30.0
    x, kx = construct_x_kx_fft(n, a, b)
    u = 1.0 / np.cosh(x)
    ux_exact = -np.tanh(x) / np.cosh(x)
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_gauss_dx_fft():
    """Test the Fourier derivative on a Gaussian function."""
    n = 128
    a, b = -10, 10
    x, kx = construct_x_kx_fft(n, a, b)
    u = np.exp(-(x**2))
    ux_exact = -2 * x * np.exp(-(x**2))
    ux_approx = dx_fft(kx, u)
    assert np.allclose(ux_exact, ux_approx)


def test_dx_fft_non_integer_order():
    """dx_fft should reject non-integer derivative orders."""
    u = np.ones(4, dtype=complex)
    kx = np.array([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(TypeError, match="must be an integer"):
        dx_fft(kx, u, n="second")


def test_dx_fft_negative_order():
    """dx_fft should reject negative derivative orders."""
    u = np.ones(4, dtype=complex)
    kx = np.array([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="non-negative"):
        dx_fft(kx, u, n=-3)
