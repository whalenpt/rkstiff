"""Tests for Chebyshev derivative computations in rkstiff.derivatives."""

import numpy as np
import pytest
from rkstiff.grids import construct_x_dx_cheb
from rkstiff import derivatives


def test_exp_trig_dx_cheb():
    """Test the Chebyshev derivative on a non-periodic function."""
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


def test_dx_cheb_shape_and_type():
    arr = np.linspace(-1, 1, 8)
    Dx = np.eye(8)
    out = derivatives.dx_cheb(arr, Dx)
    assert out.shape == arr.shape
    assert isinstance(out, np.ndarray)


def test_dx_cheb_bad_shape():
    arr = np.linspace(-1, 1, 8)
    Dx = np.eye(7)
    with pytest.raises(ValueError):
        derivatives.dx_cheb(Dx, arr)
