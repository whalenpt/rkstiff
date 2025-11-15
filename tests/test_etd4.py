import numpy as np
from rkstiff.etd import ETDConfig
from rkstiff import etd4
from rkstiff.etd4 import ETD4
from testing_util import kdv_soliton_setup, kdv_step_eval, kdv_evolve_eval


def test_etd4_step():
    """Test the ETD4 solver step method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD4(lin_op=linear_op, nl_func=nl_func)
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, 1e-6)


def test_etd4_evolve():
    """Test the ETD4 solver evolve method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD4(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-6)


def test_etd4_diagonal_basic():
    """Test _Etd4Diagonal with a small 1D diagonal operator."""
    lin_op = np.array([-1.0, -2.0], dtype=np.float64)
    nl_func = lambda u: u**2
    config = ETDConfig(modecutoff=0.01, contour_points=8, contour_radius=1.0)
    method = etd4._Etd4Diagonal(lin_op, nl_func, config)

    # Check initialization shapes
    n = lin_op.shape[0]
    assert method._EL.shape[0] == n
    assert method._NL1.shape[0] == n

    # Update coefficients for a step size
    h = 0.1
    method.update_coeffs(h)

    # Test n1_init stores nonlinear evaluation
    u0 = np.array([0.5, -0.5])
    method.n1_init(u0)
    np.testing.assert_array_equal(method._NL1, nl_func(u0))

    # Test one stage update
    u1 = method.update_stages(u0)
    assert u1.shape == u0.shape
    # FSAL: NL1 should be updated
    np.testing.assert_array_equal(method._NL1, nl_func(u1))


def test_etd4_nondiagonal_basic():
    """Test _Etd4NonDiagonal with a small 2x2 matrix operator."""
    lin_op = np.array([[-1.0, 0.1], [0.0, -2.0]])
    nl_func = lambda u: u**2
    config = ETDConfig(contour_points=4, contour_radius=1.0)
    method = etd4._Etd4NonDiagonal(lin_op, nl_func, config)

    # Initialization shapes
    n = lin_op.shape[0]
    assert method._EL.shape == lin_op.shape
    assert method._NL1.shape[0] == n

    # Update coefficients
    h = 0.05
    method.update_coeffs(h)

    # n1_init stores nonlinear evaluation
    u0 = np.array([0.1, -0.2])
    method.n1_init(u0)
    np.testing.assert_array_equal(method._NL1, nl_func(u0))

    # Stage update
    u1 = method.update_stages(u0)
    assert u1.shape == u0.shape
    np.testing.assert_array_equal(method._NL1, nl_func(u1))


def test_etd4_solver_diagonal_and_reset():
    """Test ETD4 solver with diagonal operator and reset behavior."""
    lin_op = np.array([-1.0, -0.5])
    nl_func = lambda u: u**2
    config = ETDConfig(modecutoff=0.01, contour_points=4, contour_radius=1.0)

    solver = etd4.ETD4(lin_op, nl_func, etd_config=config, loglevel="DEBUG")
    assert isinstance(solver._method, etd4._Etd4Diagonal)
    u0 = np.array([0.2, 0.3])

    # First stage update triggers n1_init
    h = 0.1
    u1 = solver._update_stages(u0, h)
    assert u1.shape == u0.shape
    # Coeffs cached
    assert solver._h_coeff == h

    # Reset solver clears initialization flag
    solver._reset()
    assert solver._h_coeff is None
    assert solver._ETD4__n1_init is False


def test_etd4_solver_nondiagonal():
    """Test ETD4 solver with full matrix operator."""
    lin_op = np.array([[-1.0, 0.1], [0.2, -0.5]])
    nl_func = lambda u: u**2
    config = ETDConfig(contour_points=4, contour_radius=1.0)

    solver = etd4.ETD4(lin_op, nl_func, etd_config=config)
    assert isinstance(solver._method, etd4._Etd4NonDiagonal)

    u0 = np.array([0.1, 0.2])
    h = 0.05
    u1 = solver._update_stages(u0, h)
    assert u1.shape == u0.shape


def test_update_coeffs_idempotent_diagonal():
    """Check that update_coeffs does not recompute for same step size."""
    lin_op = np.array([-1.0])
    nl_func = lambda u: u**2
    config = ETDConfig()
    method = etd4._Etd4Diagonal(lin_op, nl_func, config)

    h = 0.1
    method.update_coeffs(h)
    old_a21 = method._a21.copy()
    method.update_coeffs(h)
    np.testing.assert_array_equal(old_a21, method._a21)
