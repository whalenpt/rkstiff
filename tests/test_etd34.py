"""Tests for the ETD34 solver and its components."""

import numpy as np
import pytest
from rkstiff.etd34 import _Etd34Diagonal, _Etd34Diagonalized, _Etd34NonDiagonal, ETD34
from rkstiff.etd import ETDConfig, SolverConfig
from testing_util import allen_cahn_setup, kdv_soliton_setup, kdv_evolve_eval, kdv_step_eval


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


def test_etd34_nondiag():
    """Test the ETD34 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    config = SolverConfig(epsilon=1e-3)
    etd_config = ETDConfig(contour_points=64, contour_radius=20)
    solver = ETD34(lin_op=linear_op, nl_func=nl_func, config=config, etd_config=etd_config)
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_etd34_step_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD34(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1)
    )  # small epsilon -> actual steps will match requested in step method
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)


def test_etd34_evolve_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD34(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-4)
    )  # small epsilon -> actual step will match requested
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd34_diagonal_coeff_update_and_stage():
    n = 4
    lin_op = np.linspace(-1, 1, n)
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=2.0)
    diag = _Etd34Diagonal(lin_op, dummy_nl, etd_conf)
    diag.update_coeffs(0.1)
    u = np.linspace(1, 2, n, dtype=np.complex128)
    diag.n1_init(u)
    k, err = diag.update_stages(u, accept=True)
    assert k.shape == (n,)
    assert err.shape == (n,)
    assert np.all(np.isfinite(k))
    assert np.any(np.abs(err) > 0)


def test_etd34_diagonalized_basic():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=1.0)
    diagz = _Etd34Diagonalized(lin_op, dummy_nl, etd_conf)
    diagz.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    diagz.n1_init(u)
    k, err = diagz.update_stages(u, accept=True)
    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_etd34_diagonalized_bad_inputs():
    etd_conf = ETDConfig()
    singular = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        _Etd34Diagonalized(singular, dummy_nl, etd_conf)
    with pytest.raises(ValueError):
        _Etd34Diagonalized(np.array([1, 2, 3]), dummy_nl, etd_conf)


def test_etd34_nondiagonal_basic():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=1.0)
    nondiag = _Etd34NonDiagonal(lin_op, dummy_nl, etd_conf)
    nondiag.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    nondiag.n1_init(u)
    k, err = nondiag.update_stages(u, accept=True)
    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_etd34_full_solver_diagonal_mode():
    n = 8
    lin_op = np.linspace(-0.5, 0.5, n)
    etd_conf = ETDConfig(contour_points=16, contour_radius=2)
    solver = ETD34(lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3), etd_config=etd_conf)
    u0 = np.ones(n, dtype=np.complex128)
    solver._reset()
    u_final, err = solver._update_stages(u0, h=0.1)
    assert u_final.shape == u0.shape
    assert np.all(np.isfinite(u_final))
    assert err.shape == u0.shape


def test_etd34_full_solver_matrix_mode():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    etd_conf = ETDConfig(contour_points=8, contour_radius=1)
    solver = ETD34(
        lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3), etd_config=etd_conf, diagonalize=True
    )
    u0 = np.array([1.0, 0.5], dtype=np.complex128)
    solver._reset()
    u_final, err = solver._update_stages(u0, h=0.05)
    assert u_final.shape == (2,)
    assert np.all(np.isfinite(u_final))
    assert err.shape == (2,)


def test_etd34_solver_order():
    lin_op = np.array([1.0, 2.0])
    solver = ETD34(lin_op=lin_op, nl_func=dummy_nl)
    assert solver._q() == 4


def test_etd34_reset_and_coeff_update():
    lin_op = np.array([1.0, 2.0])
    solver = ETD34(lin_op=lin_op, nl_func=dummy_nl)
    solver._reset()
    solver._update_coeffs(0.1)
    solver._update_coeffs(0.1)  # Should not update again
    solver._update_coeffs(0.2)  # Should update
    assert solver._h_coeff == 0.2

def test_etd34diagonalized_large_condition_warning(caplog):
    A = np.array([[1, 1e6], [0, 1]])
    mod_logger = _Etd34Diagonalized.__module__

    with caplog.at_level("WARNING", logger=mod_logger):
        _ = _Etd34Diagonalized(A, dummy_nl, ETDConfig())

    assert any("condition number" in m for m in caplog.messages)
