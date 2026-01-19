"""Tests for the ETD4K solver."""

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator

from rkstiff.etd4k import ETD4K, _ETD4K
from rkstiff.krylov import KrylovConfig
from rkstiff.etd import ETDConfig


def dummy_nl(u):
    """Simple nonlinear function for testing."""
    return u**2


# ================================================================
#  _ETD4K Strategy Tests
# ================================================================
class TestETD4KStrategy:
    """Tests for the _ETD4K internal strategy class."""

    def test_initialization_with_dense_matrix(self):
        """Test initialization with dense numpy array."""
        A = np.diag([-1.0, -2.0, -3.0])
        config = KrylovConfig(m_init=5, m_max=10)
        method = _ETD4K(A, dummy_nl, config)

        assert method._n == 3
        assert method._NL1 is None

    def test_initialization_with_sparse_matrix(self):
        """Test initialization with sparse matrix."""
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(10, 10), dtype=float)
        config = KrylovConfig()
        method = _ETD4K(A, dummy_nl, config)

        assert method._n == 10

    def test_initialization_with_linear_operator(self):
        """Test initialization with LinearOperator."""

        def matvec(v):
            return -v

        A = LinearOperator((5, 5), matvec=matvec)
        config = KrylovConfig()
        method = _ETD4K(A, dummy_nl, config)

        assert method._n == 5

    def test_initialization_with_callable(self):
        """Test initialization with callable."""

        def A(v):
            return -2 * v

        config = KrylovConfig()
        method = _ETD4K(A, dummy_nl, config)

        # Size determined on first call
        assert method._n is None

    def test_n1_init(self):
        """Test nonlinear evaluation initialization."""
        A = np.diag([-1.0, -2.0])
        config = KrylovConfig()
        method = _ETD4K(A, dummy_nl, config)

        u0 = np.array([1.0, 2.0])
        method.n1_init(u0)

        np.testing.assert_array_equal(method._NL1, dummy_nl(u0))

    def test_update_stages_returns_correct_shape(self):
        """Test that update_stages returns correct shape."""
        n = 5
        A = np.diag(-np.ones(n))
        config = KrylovConfig(m_init=5, m_max=10, tol=1e-6)
        method = _ETD4K(A, dummy_nl, config)

        u0 = np.ones(n)
        method.n1_init(u0)
        method.update_coeffs(0.1)

        u1 = method.update_stages(u0, 0.1)

        assert u1.shape == u0.shape
        assert np.all(np.isfinite(u1))

    def test_update_stages_fsal(self):
        """Test FSAL principle - NL1 is updated after step."""
        n = 3
        A = np.diag([-1.0, -2.0, -3.0])
        config = KrylovConfig(m_init=5, tol=1e-6)
        method = _ETD4K(A, dummy_nl, config)

        u0 = np.array([1.0, 0.5, 0.25])
        method.n1_init(u0)
        method.update_coeffs(0.1)

        u1 = method.update_stages(u0, 0.1)

        # NL1 should be updated to nl_func(u1) (FSAL)
        np.testing.assert_array_almost_equal(method._NL1, dummy_nl(u1))


# ================================================================
#  ETD4K Solver Tests
# ================================================================
class TestETD4KSolver:
    """Tests for the ETD4K solver class."""

    def test_initialization_with_dense_array(self):
        """Test solver initialization with dense array."""
        A = np.diag([-1.0, -2.0])
        solver = ETD4K(A, dummy_nl)

        assert isinstance(solver._method, _ETD4K)

    def test_initialization_with_sparse_matrix(self):
        """Test solver initialization with sparse matrix."""
        A = diags([-1], [0], shape=(10, 10), dtype=float)
        solver = ETD4K(A, dummy_nl)

        assert isinstance(solver._method, _ETD4K)

    def test_initialization_with_linear_operator(self):
        """Test solver initialization with LinearOperator."""

        def matvec(v):
            return -v

        A = LinearOperator((5, 5), matvec=matvec)
        solver = ETD4K(A, dummy_nl)

        assert isinstance(solver._method, _ETD4K)

    def test_initialization_with_custom_config(self):
        """Test solver initialization with custom KrylovConfig."""
        A = np.diag([-1.0])
        krylov_config = KrylovConfig(m_init=20, m_max=50, tol=1e-8)
        solver = ETD4K(A, dummy_nl, krylov_config=krylov_config)

        assert solver.krylov_config.m_init == 20
        assert solver.krylov_config.m_max == 50
        assert solver.krylov_config.tol == 1e-8

    def test_reset(self):
        """Test solver reset functionality."""
        A = np.diag([-1.0, -2.0])
        solver = ETD4K(A, dummy_nl)

        # Set some state
        solver._h_coeff = 0.1
        solver._ETD4K__n1_init = True

        solver._reset()

        assert solver._h_coeff is None
        assert solver._ETD4K__n1_init is False

    def test_update_coeffs_caching(self):
        """Test that coefficients are cached for same step size."""
        A = np.diag([-1.0])
        solver = ETD4K(A, dummy_nl)

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
        solver = ETD4K(A, dummy_nl)

        u0 = np.ones(n)
        h = 0.1

        u1 = solver.step(u0, h)

        assert u1.shape == u0.shape
        assert np.all(np.isfinite(u1))

    def test_evolve_method(self):
        """Test evolve method over multiple steps."""
        n = 5
        A = np.diag(-np.ones(n))
        solver = ETD4K(A, dummy_nl)

        u0 = np.ones(n)
        t0, tf = 0.0, 0.5
        h = 0.1

        u_final = solver.evolve(u0, t0, tf, h, store_data=True)

        assert u_final.shape == u0.shape
        assert np.all(np.isfinite(u_final))
        assert len(solver.t) > 1
        assert len(solver.u) > 1

    def test_evolve_no_store(self):
        """Test evolve without storing intermediate data."""
        n = 3
        A = np.diag([-1.0, -2.0, -3.0])
        solver = ETD4K(A, dummy_nl)

        u0 = np.ones(n)
        u_final = solver.evolve(u0, 0, 0.5, 0.1, store_data=False)

        assert np.all(np.isfinite(u_final))
        assert len(solver.u) == 0 or len(solver.u) == 1  # May store initial


# ================================================================
#  Integration Tests
# ================================================================
class TestETD4KIntegration:
    """Integration tests for ETD4K."""

    def test_heat_equation(self):
        """Test ETD4K on 1D heat equation."""
        # u_t = u_xx with Dirichlet BCs
        n = 20
        dx = 1.0 / (n + 1)

        # Laplacian with Dirichlet BCs
        A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n), dtype=float) / dx**2

        # Zero nonlinearity for pure heat equation
        def zero_nl(u):
            return np.zeros_like(u)

        solver = ETD4K(A, zero_nl)

        # Initial condition: sin(pi*x)
        x = np.linspace(dx, 1 - dx, n)
        u0 = np.sin(np.pi * x)

        # Evolve
        u_final = solver.evolve(u0, 0, 0.01, h=0.002, store_data=False)

        # Solution should decay
        assert np.linalg.norm(u_final.real) < np.linalg.norm(u0)

    def test_linear_operator_heat_equation(self):
        """Test ETD4K with LinearOperator on heat equation."""
        n = 20
        dx = 1.0 / (n + 1)

        # Create LinearOperator for Laplacian
        def laplacian_matvec(u):
            result = np.zeros_like(u)
            result[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
            result[0] = (-2 * u[0] + u[1]) / dx**2
            result[-1] = (u[-2] - 2 * u[-1]) / dx**2
            return result

        A = LinearOperator((n, n), matvec=laplacian_matvec, dtype=float)

        def zero_nl(u):
            return np.zeros_like(u)

        solver = ETD4K(A, zero_nl)

        x = np.linspace(dx, 1 - dx, n)
        u0 = np.sin(np.pi * x)

        u_final = solver.evolve(u0, 0, 0.01, h=0.002, store_data=False)
        assert np.all(np.isfinite(u_final))

    def test_nonlinear_problem(self):
        """Test ETD4K on problem with nonlinearity."""
        n = 10
        A = np.diag(-np.ones(n))

        def cubic_nl(u):
            return -(u**3)

        solver = ETD4K(A, cubic_nl)

        u0 = np.ones(n) * 0.5
        u_final = solver.evolve(u0, 0, 0.5, h=0.1, store_data=False)

        assert np.all(np.isfinite(u_final))
        # Solution should be bounded
        assert np.max(np.abs(u_final)) < 10

    def test_callable_operator(self):
        """Test ETD4K with callable operator."""
        n = 5

        def A(v):
            return -2 * v  # Simple decay

        def zero_nl(u):
            return np.zeros_like(u)

        krylov_config = KrylovConfig(m_init=5, tol=1e-6)
        solver = ETD4K(A, zero_nl, krylov_config=krylov_config)

        u0 = np.ones(n)
        u_final = solver.evolve(u0, 0, 0.5, h=0.1, store_data=False)

        # Should decay exponentially: exp(-2*0.5) = exp(-1) ≈ 0.368
        expected_decay = np.exp(-1)
        np.testing.assert_array_almost_equal(
            u_final.real / u0, np.ones(n) * expected_decay, decimal=1  # Loose tolerance for Krylov approximation
        )


# ================================================================
#  Debug/Logging Tests
# ================================================================
class TestETD4KLogging:
    """Tests for logging behavior."""

    def test_debug_logging(self, caplog):
        """Test debug logging is produced."""
        A = np.diag([-1.0])
        solver = ETD4K(A, dummy_nl, loglevel="DEBUG")

        with caplog.at_level("DEBUG"):
            solver._update_coeffs(0.1)

        assert any("step size" in m.lower() for m in caplog.messages)
