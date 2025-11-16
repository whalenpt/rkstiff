"""Comprehensive tests for the IF4 solver and its components."""

import numpy as np
import pytest
from rkstiff.if4 import IF4, _IF4Diagonal, _IF4NonDiagonal
from testing_util import kdv_soliton_setup, kdv_step_eval, kdv_evolve_eval


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


def linear_nl(u):
    """Linear nonlinear function (returns zero) for testing pure linear evolution."""
    return np.zeros_like(u)


# ============================================================================
# Existing Tests (KdV Integration)
# ============================================================================


def test_if4_kdv_step():
    """Test IF4 single step accuracy on KdV soliton."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF4(lin_op=linear_op, nl_func=nl_func)
    kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-5)


def test_if4_kdv_evolve():
    """Test IF4 evolve method on KdV soliton."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF4(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h=h, tf=h * steps, tol=1e-5)


def test_if4_reset_and_coeff_update():
    """Test internal state reset and coefficient caching."""
    lin_op = np.array([1.0, 2.0])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl)

    # Test reset
    solver._reset()
    assert solver._IF4__n1_init == False
    assert solver._h_coeff is None

    # Test coefficient caching
    solver._update_coeffs(0.1)
    assert solver._h_coeff == 0.1

    solver._update_coeffs(0.1)  # Should not recompute
    assert solver._h_coeff == 0.1

    solver._update_coeffs(0.2)  # Should update
    assert solver._h_coeff == 0.2


# ============================================================================
# Non-Diagonal Operator Tests (Coverage for _IF4NonDiagonal)
# ============================================================================


def test_if4_non_diagonal_simple():
    """Test IF4 with a simple non-diagonal matrix operator."""
    # Simple 2x2 system: du/dt = L*u + N(u)
    # L = [[0, 1], [1, 0]] (coupling matrix)
    lin_op = np.array([[0.0, 1.0], [1.0, 0.0]])

    def nl_func(u):
        return -0.1 * u**3  # Cubic nonlinearity

    u0 = np.array([1.0, 0.5])
    h = 0.01

    solver = IF4(lin_op=lin_op, nl_func=nl_func)

    # Verify non-diagonal strategy is selected
    assert isinstance(solver._method, _IF4NonDiagonal)

    # Take one step
    u1 = solver.step(u0, h)

    # Solution should change (basic sanity check)
    assert not np.allclose(u1, u0)
    assert u1.shape == u0.shape


def test_if4_non_diagonal_matrix_exponential():
    """Verify matrix exponential computation for non-diagonal operator."""
    # Test with a known matrix exponential
    lin_op = np.array([[1.0, 0.5], [0.5, 1.0]])

    strategy = _IF4NonDiagonal(lin_op=lin_op, nl_func=dummy_nl)
    h = 0.1
    strategy.update_coeffs(h)

    # Check that exponentials were computed
    assert strategy._EL.shape == lin_op.shape
    assert strategy._EL2.shape == lin_op.shape

    # Verify exp(0) = I when h=0
    strategy.update_coeffs(0.0)
    assert np.allclose(strategy._EL, np.eye(2))
    assert np.allclose(strategy._EL2, np.eye(2))


def test_if4_non_diagonal_pure_linear():
    """Test non-diagonal operator with zero nonlinearity (pure linear evolution)."""
    # For du/dt = L*u with u0 = [1, 0], solution is u(t) = exp(L*t) * u0
    lin_op = np.array([[0.0, 2.0], [-2.0, 0.0]])  # Rotation matrix

    solver = IF4(lin_op=lin_op, nl_func=linear_nl)

    u0 = np.array([1.0, 0.0])
    h = 0.01
    n_steps = 10

    # Evolve
    u_final = u0.copy()
    for _ in range(n_steps):
        u_final = solver.step(u_final, h)

    # Compute exact solution using matrix exponential
    from scipy.linalg import expm

    u_exact = expm(lin_op * h * n_steps) @ u0

    # Should be accurate for pure linear problem
    assert np.allclose(u_final, u_exact, rtol=1e-6)


def test_if4_non_diagonal_conservation():
    """Test that non-diagonal IF4 preserves expected structure."""
    # Skew-symmetric operator (should preserve energy ||u||^2)
    lin_op = np.array([[0.0, -1.0], [1.0, 0.0]])

    def conservative_nl(u):
        # Nonlinearity that preserves structure
        return np.zeros_like(u)

    solver = IF4(lin_op=lin_op, nl_func=conservative_nl)

    u0 = np.array([1.0, 1.0])
    h = 0.01
    energy_initial = np.linalg.norm(u0) ** 2

    u = u0.copy()
    for _ in range(100):
        u = solver.step(u, h)

    energy_final = np.linalg.norm(u) ** 2

    # Energy should be conserved for skew-symmetric operator
    assert np.allclose(energy_initial, energy_final, rtol=1e-6)


# ============================================================================
# Diagonal Strategy Direct Tests
# ============================================================================


def test_if4_diagonal_direct():
    """Test _IF4Diagonal strategy directly."""
    lin_op = np.array([1.0, -2.0, 3.0])
    strategy = _IF4Diagonal(lin_op=lin_op, nl_func=dummy_nl)

    u = np.array([1.0, 2.0, 3.0])
    h = 0.1

    # Initialize coefficients
    strategy.update_coeffs(h)
    strategy.n1_init(u)

    # Perform update
    u_new = strategy.update_stages(u, h)

    # Basic sanity checks
    assert u_new.shape == u.shape
    assert not np.allclose(u_new, u)  # Should have changed


def test_if4_diagonal_fsal_property():
    """Verify First-Same-As-Last (FSAL) property for diagonal strategy."""
    lin_op = np.array([1.0, -1.0])
    strategy = _IF4Diagonal(lin_op=lin_op, nl_func=dummy_nl)

    u = np.array([1.0, 2.0])
    h = 0.01

    strategy.update_coeffs(h)
    strategy.n1_init(u)

    # Store NL1 after initialization
    nl1_before = strategy._NL1.copy()

    # Perform update (which should update NL1 for next step)
    u_new = strategy.update_stages(u, h)

    # NL1 should now be N(u_new) for FSAL
    nl1_after = strategy._NL1.copy()
    expected_nl1 = dummy_nl(u_new)

    assert np.allclose(nl1_after, expected_nl1)
    assert not np.allclose(nl1_before, nl1_after)


# ============================================================================
# Edge Cases and Input Validation
# ============================================================================


def test_if4_complex_values():
    """Test IF4 with complex-valued states (common in spectral methods)."""
    lin_op = np.array([1.0j, -2.0j, 3.0j])

    def complex_nl(u):
        return -1j * np.abs(u) ** 2 * u

    solver = IF4(lin_op=lin_op, nl_func=complex_nl)

    u0 = np.array([1.0 + 1.0j, 2.0 - 0.5j, -1.0 + 2.0j])
    h = 0.01

    u1 = solver.step(u0, h)

    assert u1.dtype == np.complex128
    assert u1.shape == u0.shape


def test_if4_zero_stepsize():
    """Test behavior with zero step size."""
    lin_op = np.array([1.0, 2.0])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl)

    u0 = np.array([1.0, 2.0])
    h = 0.0

    u1 = solver.step(u0, h)

    # With h=0, solution should not change significantly
    assert np.allclose(u1, u0, atol=1e-10)


def test_if4_large_system():
    """Test IF4 with a larger system size."""
    n = 100
    lin_op = np.linspace(-1, 1, n)

    def nl_func(u):
        return -0.01 * u**3

    solver = IF4(lin_op=lin_op, nl_func=nl_func)

    u0 = np.random.randn(n)
    h = 0.01

    u1 = solver.step(u0, h)

    assert u1.shape == (n,)
    assert not np.allclose(u1, u0)


def test_if4_stiff_problem():
    """Test IF4 on a stiff problem where eigenvalues have large negative real parts."""
    # Stiff operator with eigenvalues ranging from -1000 to -1
    lin_op = -np.logspace(0, 3, 50)

    def nl_func(u):
        return 0.1 * np.sin(u)

    solver = IF4(lin_op=lin_op, nl_func=nl_func)

    u0 = np.ones(50)
    h = 0.01  # This would be unstable for explicit RK4

    # Should remain stable due to integrating factor
    u = u0.copy()
    for _ in range(10):
        u = solver.step(u, h)

    # Solution should remain bounded
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) < 100  # Reasonable bound


# ============================================================================
# Logging Tests
# ============================================================================


def test_if4_logging_levels():
    """Test that different logging levels can be set."""
    lin_op = np.array([1.0, 2.0])

    # Test different log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        solver = IF4(lin_op=lin_op, nl_func=dummy_nl, loglevel=level)
        assert solver.logger is not None


def test_if4_logging_output(caplog):
    """Test that debug logging produces expected messages."""
    import logging

    lin_op = np.array([1.0, 2.0])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl, loglevel="DEBUG")

    with caplog.at_level(logging.DEBUG):
        u0 = np.array([1.0, 2.0])
        solver.step(u0, 0.1)
        solver.step(u0, 0.2)  # Different step size

    # Check that coefficient update was logged
    assert any("coefficients updated" in record.message.lower() for record in caplog.records)


# ============================================================================
# Comparison Tests (Diagonal vs Non-Diagonal)
# ============================================================================


def test_if4_diagonal_vs_nondiagonal_equivalence():
    """Verify diagonal and non-diagonal strategies give same results for diagonal operators."""
    # Create a diagonal operator as both array and matrix
    diag_values = np.array([1.0, -2.0, 0.5])
    lin_op_diag = diag_values
    lin_op_matrix = np.diag(diag_values)

    def nl_func(u):
        return -0.1 * u**2

    # Create both solvers
    solver_diag = IF4(lin_op=lin_op_diag, nl_func=nl_func)
    solver_matrix = IF4(lin_op=lin_op_matrix, nl_func=nl_func)

    # Verify correct strategy selection
    assert isinstance(solver_diag._method, _IF4Diagonal)
    assert isinstance(solver_matrix._method, _IF4NonDiagonal)

    # Test with same initial condition
    u0 = np.array([1.0, 2.0, -0.5])
    h = 0.01

    u_diag = u0.copy()
    u_matrix = u0.copy()

    # Take several steps
    for _ in range(10):
        u_diag = solver_diag.step(u_diag, h)
        u_matrix = solver_matrix.step(u_matrix, h)

    # Results should be very close
    assert np.allclose(u_diag, u_matrix, rtol=1e-10)


# ============================================================================
# Multiple Step Tests
# ============================================================================


def test_if4_multiple_steps_same_h():
    """Test multiple consecutive steps with same step size."""
    lin_op = np.array([1.0, -1.0, 0.5])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl)

    u = np.array([1.0, 2.0, 3.0])
    h = 0.01

    # Take multiple steps
    for _ in range(5):
        u = solver.step(u, h)

    # Coefficients should only be updated once
    assert solver._h_coeff == h


def test_if4_multiple_steps_varying_h():
    """Test multiple steps with varying step sizes."""
    lin_op = np.array([1.0, -1.0])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl)

    u = np.array([1.0, 2.0])
    step_sizes = [0.01, 0.02, 0.01, 0.03, 0.01]

    for h in step_sizes:
        u = solver.step(u, h)

    # Final coefficient should match last step size
    assert solver._h_coeff == step_sizes[-1]
    assert np.all(np.isfinite(u))


def test_if4_reset_between_simulations():
    """Test that reset properly clears state between simulations."""
    lin_op = np.array([1.0, 2.0])
    solver = IF4(lin_op=lin_op, nl_func=dummy_nl)

    # First simulation
    u1 = np.array([1.0, 2.0])
    for _ in range(5):
        u1 = solver.step(u1, 0.01)

    # Reset
    solver._reset()

    # Second simulation should give same result from same IC
    u2 = np.array([1.0, 2.0])
    for _ in range(5):
        u2 = solver.step(u2, 0.01)

    assert np.allclose(u1, u2)
