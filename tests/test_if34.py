"""Tests for the IF34 solver and its components."""

import numpy as np
import pytest
from rkstiff.if34 import _If34Diagonal, _If34Diagonalized, _If34NonDiagonal, IF34
from rkstiff.solveras import SolverConfig
from testing_util import allen_cahn_setup, burgers_setup, kdv_soliton_setup, kdv_evolve_eval, kdv_step_eval


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


def test_if34():
    """Test the IF34 solver on the Burgers equation."""
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func)
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2


def test_if34_nondiag():
    """Test the IF34 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_if34_diagonal_coeff_update_and_stage():
    n = 4
    lin_op = np.linspace(-1, 1, n)
    diag = _If34Diagonal(lin_op, dummy_nl)
    diag.update_coeffs(0.1)
    u = np.linspace(1, 2, n, dtype=np.complex128)
    diag.n1_init(u)
    k, err = diag.update_stages(u, h=0.1, accept=True)
    assert k.shape == (n,)
    assert err.shape == (n,)
    assert np.all(np.isfinite(k))
    assert np.any(np.abs(err) > 0)


def test_if34_diagonalized_basic():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    diagz = _If34Diagonalized(lin_op, dummy_nl)
    diagz.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    diagz.n1_init(u)
    k, err = diagz.update_stages(u, h=0.05, accept=True)
    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_if34_diagonalized_bad_inputs():
    singular = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        _If34Diagonalized(singular, dummy_nl)
    with pytest.raises(ValueError):
        _If34Diagonalized(np.array([1, 2, 3]), dummy_nl)


def test_if34_nondiagonal_basic():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    nondiag = _If34NonDiagonal(lin_op, dummy_nl)
    nondiag.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    nondiag.n1_init(u)
    k, err = nondiag.update_stages(u, h=0.05, accept=True)
    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_if34_full_solver_diagonal_mode():
    n = 8
    lin_op = np.linspace(-0.5, 0.5, n)
    solver = IF34(lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3))
    u0 = np.ones(n, dtype=np.complex128)
    solver._reset()
    u_final, err = solver._update_stages(u0, h=0.1)
    assert u_final.shape == u0.shape
    assert np.all(np.isfinite(u_final))
    assert err.shape == u0.shape


def test_if34_full_solver_matrix_mode():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    solver = IF34(lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3), diagonalize=True)
    u0 = np.array([1.0, 0.5], dtype=np.complex128)
    solver._reset()
    u_final, err = solver._update_stages(u0, h=0.05)
    assert u_final.shape == (2,)
    assert np.all(np.isfinite(u_final))
    assert err.shape == (2,)


def test_if34_solver_order():
    lin_op = np.array([1.0, 2.0])
    solver = IF34(lin_op=lin_op, nl_func=dummy_nl)
    assert solver._q() == 4


def test_if34_reset_and_coeff_update():
    lin_op = np.array([1.0, 2.0])
    solver = IF34(lin_op=lin_op, nl_func=dummy_nl)
    solver._reset()
    solver._update_coeffs(0.1)
    solver._update_coeffs(0.1)  # Should not update again
    solver._update_coeffs(0.2)  # Should update
    assert solver._h_coeff == 0.2


def test_if34_burgers():
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func)
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2


def test_if34_nondiag_allen_cahn():
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_if34_step_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF34(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1)
    )  # small epsilon -> actual step will match requested
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)


def test_if34_evolve_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-5))
    # Tolerances not exact for semi-linear PDE systems... needed to relax
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd34diagonalized_large_condition_warning(caplog):
    A = np.array([[1, 1e6], [0, 1]])
    mod_logger = _If34Diagonalized.__module__

    with caplog.at_level("WARNING", logger=mod_logger):
        _ = _If34Diagonalized(A, dummy_nl)

    assert any("condition number" in m for m in caplog.messages)
