import numpy as np
import pytest
from rkstiff.etd35 import _Etd35Diagonal, _Etd35Diagonalized, ETD35, SolverConfig
from rkstiff.etd import ETDConfig
from testing_util import allen_cahn_setup, kdv_soliton_setup, kdv_evolve_eval, kdv_step_eval


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


def test_etd35_step_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD35(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1)
    )  # small epsilon -> actual steps will match requested in step method
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)


def test_etd35_evolve_kdv():
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD35(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-4)
    )  # small epsilon -> actual step will match requested
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd35_nondiag():
    """Test the ETD35 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    config = SolverConfig(epsilon=1e-4)
    etd_config = ETDConfig(contour_points=32, contour_radius=10)
    solver = ETD35(lin_op=linear_op, nl_func=nl_func, config=config, etd_config=etd_config)
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_etd35diagonal_coeff_update_and_stage():
    """Test _Etd35Diagonal coefficient and stage update behavior."""
    n = 4
    lin_op = np.linspace(-1, 1, n)
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=2.0)
    diag = _Etd35Diagonal(lin_op, dummy_nl, etd_conf)

    # Exercise coefficient update (both small and large modes)
    diag.update_coeffs(0.1)

    u = np.linspace(1, 2, n, dtype=np.complex128)
    diag.new_step_init(u)
    k, err = diag.update_stages(u, accept=True)

    # Basic sanity checks
    assert k.shape == (n,)
    assert err.shape == (n,)
    assert np.all(np.isfinite(k))
    assert np.any(np.abs(err) > 0)


def test_etd35diagonalized_basic():
    """Test _Etd35Diagonalized initialization and stage updates."""
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])  # rotation matrix (well-conditioned)
    etd_conf = ETDConfig(modecutoff=0.5, contour_points=8, contour_radius=1.0)
    diagz = _Etd35Diagonalized(lin_op, dummy_nl, etd_conf)

    diagz.update_coeffs(0.05)
    u = np.array([1.0, 0.0], dtype=np.complex128)
    diagz.new_step_init(u)
    k, err = diagz.update_stages(u, accept=True)

    assert k.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(k))


def test_etd35diagonalized_bad_inputs():
    """Test that _Etd35Diagonalized raises for invalid matrices."""
    etd_conf = ETDConfig()

    # Singular matrix (non-invertible)
    singular = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        _Etd35Diagonalized(singular, dummy_nl, etd_conf)

    # 1D array cannot be diagonalized
    with pytest.raises(ValueError):
        _Etd35Diagonalized(np.array([1, 2, 3]), dummy_nl, etd_conf)


def test_etd35_full_solver_diagonal_mode():
    """Test the public ETD35 solver on a diagonal operator."""
    n = 8
    lin_op = np.linspace(-0.5, 0.5, n)
    etd_conf = ETDConfig(contour_points=16, contour_radius=2)
    solver = ETD35(lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3), etd_config=etd_conf)
    u0 = np.ones(n, dtype=np.complex128)
    u_final = solver.evolve(u0, t0=0, tf=0.1, h_init=0.01, store_data=False)
    assert u_final.shape == u0.shape
    assert np.all(np.isfinite(u_final))


def test_etd35_full_solver_matrix_mode():
    """Test the ETD35 solver with a 2x2 matrix (nondiagonal)."""
    lin_op = np.array([[0.0, 1.0], [-1.0, 0.0]])
    etd_conf = ETDConfig(contour_points=8, contour_radius=1)
    solver = ETD35(lin_op=lin_op, nl_func=dummy_nl, config=SolverConfig(epsilon=1e-3), etd_config=etd_conf)
    u0 = np.array([1.0, 0.5], dtype=np.complex128)
    u_final = solver.evolve(u0, t0=0, tf=0.05, h_init=0.01, store_data=False)
    assert u_final.shape == (2,)
    assert np.all(np.isfinite(u_final))
