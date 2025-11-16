""" Extra tests for rkstiff.derivatives. """

import numpy as np
import pytest
from rkstiff.derivatives import dx_rfft, dx_fft, dx_cheb

def test_dx_rfft_empty_array():
    kx = np.array([])
    u = np.array([])
    out = dx_rfft(kx, u)
    assert out.size == 0

def test_dx_rfft_shape_mismatch():
    kx = np.array([0.0, 1.0])
    u = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        dx_rfft(kx, u)

def test_dx_fft_shape_mismatch():
    kx = np.array([0.0, 1.0])
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        dx_fft(kx, u)

def test_dx_fft_zero_derivative():
    kx = np.array([0.0, 1.0, 2.0])
    u = np.array([1.0, 2.0, 3.0])
    out = dx_fft(kx, u, n=0)
    np.testing.assert_array_equal(out, u)

def test_dx_fft_negative_n():
    kx = np.array([0.0, 1.0, 2.0])
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        dx_fft(kx, u, n=-1)

def test_dx_fft_non_integer_n():
    kx = np.array([0.0, 1.0, 2.0])
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        dx_fft(kx, u, n=1.5)

def test_dx_cheb_zero_derivative():
    D = np.eye(3)
    u = np.array([1.0, 2.0, 3.0])
    out = dx_cheb(D, u, n=0)
    np.testing.assert_array_equal(out, u)

def test_dx_cheb_bad_matrix_shape():
    D = np.eye(3, 4)
    u = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        dx_cheb(D, u)

def test_dx_cheb_bad_u_shape():
    D = np.eye(3)
    u = np.ones((2, 3, 4))
    with pytest.raises(ValueError):
        dx_cheb(D, u)

def test_dx_cheb_matrix_power():
    D = np.eye(3)
    u = np.array([1.0, 2.0, 3.0])
    out = dx_cheb(D, u, n=2)
    np.testing.assert_array_equal(out, u)
