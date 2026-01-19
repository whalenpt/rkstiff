"""Tests for the ETD35K adaptive solver."""

import numpy as np
import pytest
import warnings
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator

from rkstiff.etd35k import ETD35K, _ETD35K
from rkstiff.krylov import KrylovConfig
from rkstiff.etd import ETDConfig
from rkstiff.solveras import SolverConfig, BaseSolverAS


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


# ================================================================
#  _ETD35K Strategy Tests
# ================================================================
class TestETD35KStrategy:
    """Tests for the _ETD35K internal strategy class."""

    def test_initialization_with_dense_matrix(self):
        """Test initialization with dense numpy array."""
        A = np.diag([-1.0, -2.0, -3.0])
        config = KrylovConfig(m_init=5, m_max=10)
        method = _ETD35K(A, dummy_nl, config)

        assert method._n == 3
        assert method._NL1 is None

    def test_initialization_with_sparse_matrix(self):
        """Test initialization with sparse matrix."""
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(10, 10), dtype=float)
        config = KrylovConfig()
        method = _ETD35K(A, dummy_nl, config)

        assert method._n == 10

    def test_initialization_with_linear_operator(self):
        """Test initialization with LinearOperator."""

        def matvec(v):
            return -v

        A = LinearOperator((5, 5), matvec=matvec)
        config = KrylovConfig()
        method = _ETD35K(A, dummy_nl, config)

        assert method._n == 5

    def test_stage_init(self):
        """Test stage initialization."""
        A = np.diag([-1.0, -2.0])
        config = KrylovConfig()
        method = _ETD35K(A, dummy_nl, config)

        u0 = np.array([1.0, 2.0])
        method.stage_init(u0)

        np.testing.assert_array_equal(method._NL1, dummy_nl(u0))

    def test_update_stages_returns_tuple(self):
        """Test that update_stages returns (solution, error) tuple."""
        n = 5
        A = np.diag(-np.ones(n))
        config = KrylovConfig(m_init=5, m_max=10, tol=1e-6)
        method = _ETD35K(A, dummy_nl, config)

        u0 = np.ones(n)
        method.stage_init(u0)
        method.update_coeffs(0.1)

        result = method.update_stages(u0, accept=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        u1, err = result
        assert u1.shape == u0.shape
        assert err.shape == u0.shape
        assert np.all(np.isfinite(u1))
        assert np.all(np.isfinite(err))

    def test_update_stages_fsal_accept_true(self):
        """Test FSAL with accept=True triggers stage_init."""
        n = 3
        A = np.diag([-1.0, -2.0, -3.0])
        config = KrylovConfig(m_init=5, tol=1e-6)
        method = _ETD35K(A, dummy_nl, config)

        u0 = np.array([1.0, 0.5, 0.25])
        method.stage_init(u0)
        method.update_coeffs(0.1)

        # First call with accept=True should reinitialize NL1
        u1, err1 = method.update_stages(u0, accept=True)
        assert np.all(np.isfinite(u1))

    def test_update_stages_fsal_accept_false(self):
        """Test FSAL with accept=False skips stage_init."""
        n = 3
        A = np.diag([-1.0, -2.0, -3.0])
        config = KrylovConfig(m_init=5, tol=1e-6)
        method = _ETD35K(A, dummy_nl, config)

        u0 = np.array([1.0, 0.5, 0.25])
        method.stage_init(u0)
        method.update_coeffs(0.1)

        # Call with accept=False
        u1, err1 = method.update_stages(u0, accept=False)
        assert np.all(np.isfinite(u1))


# ================================================================
#  ETD35K Solver Tests
# ================================================================
class TestETD35KSolver:
    """Tests for the ETD35K solver class."""

    def test_initialization_with_dense_array(self):
        """Test solver initialization with dense array."""
        A = np.diag([-1.0, -2.0])
        solver = ETD35K(A, dummy_nl)

        assert isinstance(solver._method, _ETD35K)

    def test_initialization_with_sparse_matrix(self):
        """Test solver initialization with sparse matrix."""
        A = diags([-1], [0], shape=(10, 10), dtype=float)
        solver = ETD35K(A, dummy_nl)

        assert isinstance(solver._method, _ETD35K)

    def test_initialization_with_linear_operator(self):
        """Test solver initialization with LinearOperator."""

        def matvec(v):
            return -v

        A = LinearOperator((5, 5), matvec=matvec)
        solver = ETD35K(A, dummy_nl)

        assert isinstance(solver._method, _ETD35K)

    def test_initialization_with_custom_configs(self):
        """Test solver initialization with custom configurations."""
        A = np.diag([-1.0])
        solver_config = SolverConfig(epsilon=1e-6)
        krylov_config = KrylovConfig(m_init=20, m_max=50, tol=1e-8)

        solver = ETD35K(A, dummy_nl, config=solver_config, krylov_config=krylov_config)

        assert solver.config.epsilon == 1e-6
        assert solver.krylov_config.m_init == 20

    def test_reset(self):
        """Test solver reset functionality."""
        A = np.diag([-1.0, -2.0])
        solver = ETD35K(A, dummy_nl)

        # Set some state
        solver._h_coeff = 0.1
        solver._stages_init = True
        solver._accept = True

        solver._reset()

        assert solver._h_coeff is None
        assert solver._stages_init is False
        assert solver._accept is False

    def test_q_method(self):
        """Test _q method returns correct order."""
        A = np.diag([-1.0])
        solver = ETD35K(A, dummy_nl)

        # ETD35 is 5th order with 3rd order embedding, q = 4
        assert solver._q() == 4

    def test_update_coeffs_caching(self):
        """Test that coefficients are cached for same step size."""
        A = np.diag([-1.0])
        solver = ETD35K(A, dummy_nl)

        solver._update_coeffs(0.1)
        assert solver._h_coeff == 0.1

        # Same step size should not change anything
        solver._update_coeffs(0.1)
        assert solver._h_coeff == 0.1

        # Different step size should update
        solver._update_coeffs(0.2)
        assert solver._h_coeff == 0.2

    def test_step_method(self):
        """Test single step method."""
        n = 5
        A = np.diag(-np.arange(1, n + 1, dtype=float))
        config = SolverConfig(epsilon=1e-2)  # Loose tolerance
        solver = ETD35K(A, dummy_nl, config=config)

        u0 = np.ones(n)
        h = 0.1

        u1, h_used, h_suggest = solver.step(u0, h)

        assert u1.shape == u0.shape
        assert np.all(np.isfinite(u1))
        assert h_used > 0
        assert h_suggest > 0

    def test_evolve_method(self):
        """Test evolve method with adaptive stepping."""
        n = 5
        A = np.diag(-np.ones(n))
        config = SolverConfig(epsilon=1e-4)
        solver = ETD35K(A, dummy_nl, config=config)

        u0 = np.ones(n) * 0.5
        t0, tf = 0.0, 0.5

        u_final = solver.evolve(u0, t0, tf, store_data=True)

        assert u_final.shape == u0.shape
        assert np.all(np.isfinite(u_final))
        assert len(solver.t) > 1
        assert len(solver.u) > 1

    def test_evolve_no_store(self):
        """Test evolve without storing intermediate data."""
        n = 3
        A = np.diag([-1.0, -2.0, -3.0])
        solver = ETD35K(A, dummy_nl)

        u0 = np.ones(n) * 0.5
        u_final = solver.evolve(u0, 0, 0.5, store_data=False)

        assert np.all(np.isfinite(u_final))


# ================================================================
#  Adaptive Stepping Tests
# ================================================================
class TestETD35KAdaptive:
    """Tests for adaptive stepping behavior."""

    def test_step_acceptance(self):
        """Test that steps are accepted with appropriate error."""
        n = 5
        A = np.diag(-np.ones(n))
        config = SolverConfig(epsilon=1e-2)  # Loose tolerance
        solver = ETD35K(A, dummy_nl, config=config)

        u0 = np.ones(n) * 0.5
        h = 0.1

        u1, h_used, h_suggest = solver.step(u0, h)

        # Step should be accepted
        assert solver._accept is True
        assert h_used > 0

    def test_explosive_nl_minimum_step(self):
        """Test explosive nonlinearity leads to MinimumStepReached."""

        def explosive_nl(u):
            return 1e10 * u

        A = np.diag([-1.0])
        config = SolverConfig(epsilon=1e-8, minh=1e-14)
        solver = ETD35K(A, explosive_nl, config=config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(BaseSolverAS.MinimumStepReached):
                solver.evolve(np.array([1.0]), t0=0, tf=1e-3, h_init=1e-2, store_data=False)

    def test_zero_interval(self):
        """Test evolve with zero time interval."""
        A = np.diag([-1.0])
        solver = ETD35K(A, dummy_nl)
        u0 = np.array([0.5])

        result = solver.evolve(u0, 0, 0, 0.1, store_data=False)
        np.testing.assert_array_almost_equal(result, u0)

    def test_negative_interval(self):
        """Test evolve with tf < t0."""
        A = np.diag([-1.0])
        solver = ETD35K(A, dummy_nl)
        u0 = np.array([0.5])

        result = solver.evolve(u0, 1.0, 0.0, 0.1, store_data=False)
        np.testing.assert_array_almost_equal(result, u0)


# ================================================================
#  Integration Tests
# ================================================================
class TestETD35KIntegration:
    """Integration tests for ETD35K."""

    def test_heat_equation(self):
        """Test ETD35K on 1D heat equation."""
        n = 20
        dx = 1.0 / (n + 1)

        # Laplacian with Dirichlet BCs
        A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n), dtype=float) / dx**2

        def zero_nl(u):
            return np.zeros_like(u)

        config = SolverConfig(epsilon=1e-4)
        solver = ETD35K(A, zero_nl, config=config)

        # Initial condition
        x = np.linspace(dx, 1 - dx, n)
        u0 = np.sin(np.pi * x)

        # Evolve
        u_final = solver.evolve(u0, 0, 0.01, store_data=False)

        # Solution should decay
        assert np.linalg.norm(u_final.real) < np.linalg.norm(u0)

    def test_linear_operator_problem(self):
        """Test ETD35K with LinearOperator."""
        n = 15

        def matvec(u):
            # Simple decay operator
            return -np.arange(1, n + 1, dtype=float) * u

        A = LinearOperator((n, n), matvec=matvec, dtype=float)

        def zero_nl(u):
            return np.zeros_like(u)

        config = SolverConfig(epsilon=1e-4)
        solver = ETD35K(A, zero_nl, config=config)

        u0 = np.ones(n)
        u_final = solver.evolve(u0, 0, 0.5, store_data=False)

        assert np.all(np.isfinite(u_final))
        # Higher modes should decay faster
        assert np.abs(u_final.real[-1]) < np.abs(u_final.real[0])

    def test_nonlinear_reaction_diffusion(self):
        """Test ETD35K on nonlinear reaction-diffusion problem."""
        n = 15
        dx = 1.0 / (n + 1)

        # Diffusion operator
        A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n), dtype=float) / dx**2

        # Cubic nonlinearity (bistable)
        def cubic_nl(u):
            return u * (1 - u) * (u - 0.5)

        config = SolverConfig(epsilon=1e-4)
        solver = ETD35K(A, cubic_nl, config=config)

        # Initial condition
        u0 = np.ones(n) * 0.6

        u_final = solver.evolve(u0, 0, 0.1, store_data=False)

        assert np.all(np.isfinite(u_final))
        assert np.max(np.abs(u_final)) < 10  # Bounded solution

    def test_callable_operator(self):
        """Test ETD35K with callable operator."""
        n = 5

        def A(v):
            return -2 * v  # Simple decay

        def zero_nl(u):
            return np.zeros_like(u)

        config = SolverConfig(epsilon=1e-4)
        krylov_config = KrylovConfig(m_init=5, tol=1e-6)
        solver = ETD35K(A, zero_nl, config=config, krylov_config=krylov_config)

        u0 = np.ones(n)
        u_final = solver.evolve(u0, 0, 0.5, store_data=False)

        # Should decay exponentially
        expected_decay = np.exp(-1)  # exp(-2 * 0.5)
        np.testing.assert_array_almost_equal(u_final.real / u0, np.ones(n) * expected_decay, decimal=1)


# ================================================================
#  FSAL and State Tests
# ================================================================
class TestETD35KFSAL:
    """Tests for FSAL (First Same As Last) behavior."""

    def test_fsal_path(self):
        """Test FSAL principle across multiple steps."""
        A = np.diag([-1.0, -2.0])
        config = SolverConfig(epsilon=1e-2)
        solver = ETD35K(A, dummy_nl, config=config)

        u = np.array([1.0, 0.5], dtype=np.complex128)

        # First step
        solver._accept = False
        solver._stages_init = False
        u1, err1 = solver._update_stages(u, 0.1)

        # Second step with acceptance
        solver._accept = True
        u2, err2 = solver._update_stages(u1, 0.1)

        assert np.all(np.isfinite(u1))
        assert np.all(np.isfinite(u2))


# ================================================================
#  Debug/Logging Tests
# ================================================================
class TestETD35KLogging:
    """Tests for logging behavior."""

    def test_debug_logging(self, caplog):
        """Test debug logging is produced."""
        A = np.diag([-1.0])
        solver = ETD35K(A, dummy_nl, loglevel="DEBUG")

        with caplog.at_level("DEBUG"):
            solver._update_coeffs(0.1)

        assert any("step size" in m.lower() for m in caplog.messages)
