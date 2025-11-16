"""Tests for the ETD35 solver and its internal components."""

import numpy as np
import pytest
import warnings
from rkstiff.etd35 import (
    ETD35,
    _Etd35Diagonal,
    _Etd35Diagonalized,
    _Etd35NonDiagonal,
)
from rkstiff.solveras import BaseSolverAS
from rkstiff.etd import ETDConfig, SolverConfig
from testing_util import allen_cahn_setup, kdv_soliton_setup, kdv_evolve_eval, kdv_step_eval


def dummy_nl(u):
    return u**2


# ================================================================
#  REAL PDE TESTS (correctness)
# ================================================================
def test_etd35_step_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD35(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1))
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)


def test_etd35_evolve_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD35(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-4))
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd35_nondiag_allencahn():
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    solver = ETD35(
        lin_op=linear_op,
        nl_func=nl_func,
        config=SolverConfig(epsilon=1e-4),
        etd_config=ETDConfig(contour_points=32, contour_radius=10),
    )
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


# ================================================================
#  BAD INPUTS
# ================================================================
def test_etd35diagonalized_bad_inputs():
    conf = ETDConfig()
    singular = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        _Etd35Diagonalized(singular, dummy_nl, conf)

    with pytest.raises(ValueError):
        _Etd35Diagonalized(np.array([1, 2, 3]), dummy_nl, conf)


# ================================================================
#  INTERNAL STATE / COEFFS
# ================================================================
def test_etd35_reset_flags():
    lin = np.array([0.0, 1.0])
    solver = ETD35(lin, dummy_nl)
    solver._h_coeff = 0.123
    solver._stages_init = True
    solver._accept = True
    solver._reset()
    assert solver._h_coeff is None
    assert solver._stages_init is False
    assert solver._accept is False


def test_etd35_update_coeffs_skip_and_update():
    lin = np.array([1.0])
    solver = ETD35(lin, dummy_nl)
    solver._update_coeffs(0.1)
    first = solver._h_coeff
    solver._update_coeffs(0.1)
    assert solver._h_coeff == first
    solver._update_coeffs(0.2)
    assert solver._h_coeff == 0.2


# ================================================================
#  FSAL LOGIC
# ================================================================
def test_etd35_fsal_path():
    lin = np.array([0.1, -0.2])
    solver = ETD35(lin, dummy_nl, config=SolverConfig(epsilon=1e-3))
    u = np.array([1.0, 2.0], dtype=np.complex128)

    solver._accept = False
    solver._stages_init = False
    u1, _ = solver._update_stages(u, 0.1)

    solver._accept = True
    u2, _ = solver._update_stages(u1, 0.1)

    assert np.all(np.isfinite(u1))
    assert np.all(np.isfinite(u2))


# ================================================================
#  ADAPTIVE EDGE CASES
# ================================================================
def test_etd35_evolve_reject_cycle():
    """Explosive NL should cause MinimumStepReached."""

    def explosive_nl(u):
        return 1e12 * u

    solver = ETD35(np.array([-1.0]), explosive_nl, config=SolverConfig(epsilon=1e-6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(BaseSolverAS.MinimumStepReached):
            solver.evolve(np.array([1.0]), t0=0, tf=1e-3, h_init=1e-2, store_data=False)


def test_etd35_evolve_negative_tf():
    solver = ETD35(np.array([1.0]), dummy_nl)
    u0 = np.array([2.0])
    assert np.allclose(solver.evolve(u0, 1.0, 0.0, 0.1, False), u0)


def test_etd35_evolve_zero_interval():
    solver = ETD35(np.array([1.0]), dummy_nl)
    u0 = np.array([0.5])
    assert np.allclose(solver.evolve(u0, 0, 0, 0.1, False), u0)


# ================================================================
#  DIAGONAL MODE: SMALL + LARGE MODES
# ================================================================
def test_etd35diagonal_small_modes_only():
    n = 4
    lin = np.array([1e-10, 2e-10, -3e-10, -1e-10])
    diag = _Etd35Diagonal(lin, dummy_nl, ETDConfig(modecutoff=1e-8, contour_points=4, contour_radius=1))
    diag.update_coeffs(1.0)
    diag.stage_init(np.ones(n, dtype=np.complex128))
    k, err = diag.update_stages(np.ones(n, dtype=np.complex128), accept=True)
    assert np.all(np.isfinite(k))
    assert np.any(np.abs(err) > 0)


def test_etd35diagonal_large_modes_only():
    n = 4
    lin = np.array([-50.0, -20.0, 10.0, 100.0])
    diag = _Etd35Diagonal(lin, dummy_nl, ETDConfig(modecutoff=0.1))
    diag.update_coeffs(0.1)
    diag.stage_init(np.linspace(1, 2, n, dtype=np.complex128))
    k, err = diag.update_stages(np.linspace(1, 2, n, dtype=np.complex128), accept=True)
    assert np.all(np.isfinite(k))


# ================================================================
#  DIAGONALIZED: CONDITION NUMBER WARNING
# ================================================================
def test_etd35diagonalized_large_condition_warning(caplog):
    A = np.array([[1, 1e6], [0, 1]])
    mod_logger = _Etd35Diagonalized.__module__

    with caplog.at_level("WARNING", logger=mod_logger):
        _ = _Etd35Diagonalized(A, dummy_nl, ETDConfig())

    assert any("condition number" in m for m in caplog.messages)


# ================================================================
#  NON-DIAGONAL SOLVER
# ================================================================
def test_etd35_nondiagonal_coeff_shapes():
    A = np.array([[0, 1], [-1, 0]])
    solver = _Etd35NonDiagonal(A, dummy_nl, ETDConfig(contour_points=4, contour_radius=2))
    solver.update_coeffs(0.1)
    assert solver._EL.shape == (2, 2)
    assert solver._a21.shape == (2, 2)


def test_etd35_nondiagonal_stage():
    A = np.array([[0, 1], [-1, 0]])
    solver = _Etd35NonDiagonal(A, dummy_nl, ETDConfig(contour_points=4, contour_radius=1))
    solver.update_coeffs(0.1)
    solver.stage_init(np.array([1.0, 0.0], dtype=np.complex128))
    k, err = solver.update_stages(np.array([1.0, 0.0], dtype=np.complex128), accept=True)
    assert np.all(np.isfinite(k))
    assert np.all(np.isfinite(err))


# ================================================================
#  PUBLIC INTERFACE BRANCHING
# ================================================================
def test_etd35_initializes_diagonal_method():
    solver = ETD35(np.array([1.0, -1.0]), dummy_nl)
    assert isinstance(solver._method, _Etd35Diagonal)


def test_etd35_initializes_diagonal_despite_diagonalize_flag():
    """Cover branch where diag=True overrides diagonalize=True."""
    solver = ETD35(np.array([1.0, -2.0, 3.0]), dummy_nl, diagonalize=True)
    assert isinstance(solver._method, _Etd35Diagonal)


def test_etd35_initializes_diagonalized_method():
    A = np.array([[0, 1], [-1, 0]])
    solver = ETD35(A, dummy_nl, diagonalize=True)
    assert isinstance(solver._method, _Etd35Diagonalized)


# ================================================================
#  COVERAGE: DEBUG LOG IN _update_coeffs
# ================================================================
def test_etd35_update_coeffs_debug_logging(caplog):
    solver = ETD35(np.array([1.0]), dummy_nl, loglevel="DEBUG")
    with caplog.at_level("DEBUG"):
        solver._update_coeffs(0.1)
    assert any("coefficients updated" in m for m in caplog.messages)


# ================================================================
#  COVERAGE: FIRST-STEP PATH IN _update_stages
# ================================================================
def test_etd35_update_stages_first_step_path():
    lin = np.array([0.5, -0.25])
    solver = ETD35(lin, dummy_nl)

    solver._stages_init = False
    solver._accept = False

    u = np.array([1.0, 2.0], dtype=np.complex128)
    k, err = solver._update_stages(u, 0.1)

    assert solver._stages_init is True
    assert np.all(np.isfinite(k))
    assert np.all(np.isfinite(err))


def test_etd35_diagonalized_basic():
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=1.0)
    diagz = _Etd35Diagonalized(lin_op, dummy_nl, etd_conf)
    diagz.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    diagz.stage_init(u)
    k, err = diagz.update_stages(u, accept=True)
    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_etd35_diagonalized_bad_inputs():
    etd_conf = ETDConfig()
    singular = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        _Etd35Diagonalized(singular, dummy_nl, etd_conf)
    with pytest.raises(ValueError):
        _Etd35Diagonalized(np.array([1, 2, 3]), dummy_nl, etd_conf)


def test_etd35diagonalized_fsal_branch():
    """Ensure the FSAL branch (accept=True → stage_init called) is executed."""
    lin = np.array([[0.0, 1.0],
                    [-1.0, 0.0]], dtype=float)

    conf = ETDConfig(modecutoff=0.1, contour_points=4, contour_radius=1.0)
    solver = _Etd35Diagonalized(lin, dummy_nl, conf)

    # Step-size coefficients
    solver.update_coeffs(0.1)

    u = np.array([1.0, 0.0], dtype=np.complex128)

    # First call: accept=False (NO FSAL)
    solver.stage_init(u)
    k1, err1 = solver.update_stages(u, accept=False)
    assert np.all(np.isfinite(k1))
    assert np.all(np.isfinite(err1))

    # Second call: accept=True triggers FSAL → stage_init(u)
    k2, err2 = solver.update_stages(u, accept=True)
    assert np.all(np.isfinite(k2))
    assert np.all(np.isfinite(err2))
