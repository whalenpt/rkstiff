"""Tests for rkstiff.etd5 module"""

import numpy as np
import pytest
from rkstiff.etd import ETDConfig
from rkstiff.etd5 import ETD5, _Etd5Diagonal, _Etd5NonDiagonal
from testing_util import kdv_soliton_setup, kdv_evolve_eval, kdv_step_eval


def test_etd5_step():
    """Test the ETD5 solver step method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD5(lin_op=linear_op, nl_func=nl_func)
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-6)


def test_etd5_evolve():
    """Test the ETD5 solver evolve method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD5(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h=h, tf=h * steps, tol=1e-6)


# ---------------------------
# Helper functions
# ---------------------------
def nl_func(u):
    return u**2


# Small diagonal operator for testing
diag_op = np.array([0.1, -0.2], dtype=np.float64)
# Small 2x2 non-diagonal operator
non_diag_op = np.array([[0.1, 0.05], [-0.05, -0.2]], dtype=np.float64)


# ---------------------------
# Tests for _Etd5Diagonal
# ---------------------------
def test_etd5_diagonal_update_coeffs_real():
    method = _Etd5Diagonal(diag_op, nl_func, ETDConfig(modecutoff=1e-8))
    method.update_coeffs(h=0.01)
    assert np.iscomplexobj(method._a21)
    assert method._a21.shape == diag_op.shape


def test_etd5_diagonal_stages():
    method = _Etd5Diagonal(diag_op, nl_func)
    u = np.array([1.0, 2.0], dtype=np.float64)
    method.update_coeffs(h=0.1)
    method.n1_init(u)
    u_next = method.update_stages(u)
    assert u_next.shape == u.shape
    assert np.all(np.isfinite(u_next))


# ---------------------------
# Tests for _Etd5NonDiagonal
# ---------------------------
def test_etd5_nondiagonal_update_coeffs_and_stages():
    method = _Etd5NonDiagonal(non_diag_op, nl_func, ETDConfig(contour_points=4, contour_radius=0.5))
    # Test coefficient update
    method.update_coeffs(0.01)
    assert method._EL.shape == non_diag_op.shape
    assert np.iscomplexobj(method._EL)
    # Test stages
    u = np.array([1.0, 2.0], dtype=np.float64)
    method.n1_init(u)
    u_next = method.update_stages(u)
    assert u_next.shape == u.shape
    assert np.all(np.isfinite(u_next))


# ---------------------------
# Tests for ETD5 wrapper
# ---------------------------
@pytest.mark.parametrize("lin_op,diag", [(diag_op, True), (non_diag_op, False)])
def test_etd5_wrapper(lin_op, diag):
    solver = ETD5(lin_op, nl_func)
    u = np.array([1.0, 2.0])
    u_next = solver._update_stages(u, h=0.01)
    # Check output shape
    assert u_next.shape == u.shape
    # Check FSAL principle
    if diag:
        assert np.allclose(solver._method._NL1, solver._method._NL6)
