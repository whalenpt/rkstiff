"""Tests for the Krylov subspace methods (KIOPS) module."""

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import expm

from rkstiff.krylov import (
    KrylovConfig,
    KrylovStats,
    arnoldi_step,
    phiv_dense,
    kiops,
    phi_functions,
    _apply_operator,
)


# ================================================================
#  KrylovConfig Tests
# ================================================================
class TestKrylovConfig:
    """Tests for KrylovConfig parameter validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KrylovConfig()
        assert config.m_init == 10
        assert config.m_max == 128
        assert config.tol == 1e-10
        assert config.p == 2
        assert config.task1_norm == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = KrylovConfig(m_init=20, m_max=64, tol=1e-8, p=3, task1_norm=1)
        assert config.m_init == 20
        assert config.m_max == 64
        assert config.tol == 1e-8
        assert config.p == 3
        assert config.task1_norm == 1

    def test_m_init_validation(self):
        """Test m_init must be positive integer."""
        with pytest.raises(ValueError):
            KrylovConfig(m_init=0)
        with pytest.raises(ValueError):
            KrylovConfig(m_init=-1)
        with pytest.raises(TypeError):
            KrylovConfig(m_init=5.5)

    def test_m_max_validation(self):
        """Test m_max must be positive integer."""
        with pytest.raises(ValueError):
            KrylovConfig(m_max=0)
        with pytest.raises(ValueError):
            KrylovConfig(m_max=-5)
        with pytest.raises(TypeError):
            KrylovConfig(m_max=10.0)

    def test_tol_validation(self):
        """Test tol must be positive."""
        with pytest.raises(ValueError):
            KrylovConfig(tol=0)
        with pytest.raises(ValueError):
            KrylovConfig(tol=-1e-10)

    def test_p_validation(self):
        """Test p must be non-negative integer."""
        with pytest.raises(ValueError):
            KrylovConfig(p=-1)
        with pytest.raises(TypeError):
            KrylovConfig(p=2.5)
        # p=0 should be valid (full orthogonalization)
        config = KrylovConfig(p=0)
        assert config.p == 0

    def test_task1_norm_validation(self):
        """Test task1_norm must be 1, 2, or inf."""
        with pytest.raises(ValueError):
            KrylovConfig(task1_norm=3)
        with pytest.raises(ValueError):
            KrylovConfig(task1_norm=0)
        # Valid norms
        KrylovConfig(task1_norm=1)
        KrylovConfig(task1_norm=2)
        KrylovConfig(task1_norm=np.inf)


# ================================================================
#  Helper Function Tests
# ================================================================
class TestApplyOperator:
    """Tests for _apply_operator helper."""

    def test_dense_matrix(self):
        """Test with dense numpy array."""
        A = np.array([[1, 2], [3, 4]])
        v = np.array([1, 0])
        result = _apply_operator(A, v)
        np.testing.assert_array_equal(result, [1, 3])

    def test_sparse_matrix(self):
        """Test with sparse matrix."""
        A = diags([1, 2, 3], 0, dtype=float)
        v = np.array([1, 1, 1])
        result = _apply_operator(A, v)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_linear_operator(self):
        """Test with LinearOperator."""
        def matvec(v):
            return 2 * v
        A = LinearOperator((3, 3), matvec=matvec)
        v = np.array([1, 2, 3])
        result = _apply_operator(A, v)
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_callable(self):
        """Test with callable."""
        def A(v):
            return -v
        v = np.array([1, 2, 3])
        result = _apply_operator(A, v)
        np.testing.assert_array_equal(result, [-1, -2, -3])


# ================================================================
#  Arnoldi Step Tests
# ================================================================
class TestArnoldiStep:
    """Tests for arnoldi_step function."""

    def test_basic_arnoldi(self):
        """Test basic Arnoldi iteration."""
        n = 5
        A = np.diag(np.arange(1, n + 1, dtype=float))
        m = 3

        V = np.zeros((n, m + 1))
        H = np.zeros((m + 2, m + 1))

        # Initial vector
        v0 = np.ones(n) / np.sqrt(n)
        V[:, 0] = v0

        # Perform Arnoldi steps
        for j in range(m):
            V, H, beta, breakdown = arnoldi_step(A, V, H, j, p=0)
            assert not breakdown
            assert beta > 0

        # Check orthogonality of V
        Vm = V[:, :m]
        VtV = Vm.T @ Vm
        np.testing.assert_array_almost_equal(VtV, np.eye(m), decimal=10)

    def test_incomplete_orthogonalization(self):
        """Test Arnoldi with incomplete orthogonalization (p > 0)."""
        n = 10
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Symmetric
        m = 5

        V = np.zeros((n, m + 1))
        H = np.zeros((m + 2, m + 1))
        V[:, 0] = np.random.randn(n)
        V[:, 0] /= np.linalg.norm(V[:, 0])

        for j in range(m):
            V, H, beta, breakdown = arnoldi_step(A, V, H, j, p=2)

        # H should be upper Hessenberg
        assert np.all(np.isfinite(H[:m + 1, :m]))

    def test_breakdown_detection(self):
        """Test that breakdown is detected for invariant subspace."""
        # Create operator with known invariant subspace
        n = 4
        A = np.zeros((n, n))
        A[0, 0] = 1.0  # Only first component is nonzero

        V = np.zeros((n, 3))
        H = np.zeros((4, 3))
        V[:, 0] = np.array([1, 0, 0, 0])

        V, H, beta, breakdown = arnoldi_step(A, V, H, 0, p=0)
        # Should detect breakdown since Av is in span of v
        assert breakdown or beta < 1e-10


# ================================================================
#  phiv_dense Tests
# ================================================================
class TestPhivDense:
    """Tests for phiv_dense function."""

    def test_phi0_is_exp(self):
        """Test that phi_0(H)*e_1 = exp(H)*e_1."""
        H = np.array([[0, 1], [-1, 0]], dtype=float)  # Rotation matrix generator
        result = phiv_dense(H, 0)
        expected = expm(H)[:, 0:1]
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_phi_values_small_matrix(self):
        """Test phi values for a small diagonal matrix."""
        H = np.diag([-0.5, -1.0])
        result = phiv_dense(H, 2)

        # phi_0 = exp
        np.testing.assert_array_almost_equal(
            result[:, 0],
            expm(H)[:, 0],
            decimal=10
        )

    def test_phi_multiple_functions(self):
        """Test computing multiple phi functions."""
        H = np.diag([0.1, -0.1])
        result = phiv_dense(H, 3)
        assert result.shape == (2, 4)  # phi_0, phi_1, phi_2, phi_3


# ================================================================
#  phi_functions Tests (scalar/array)
# ================================================================
class TestPhiFunctions:
    """Tests for phi_functions (element-wise)."""

    def test_phi0_is_exp(self):
        """Test phi_0 = exp."""
        z = np.array([0, 1, -1, 0.5])
        result = phi_functions(z, 0)
        np.testing.assert_array_almost_equal(result[..., 0], np.exp(z), decimal=12)

    def test_phi1_formula(self):
        """Test phi_1 = (exp(z) - 1) / z."""
        z = np.array([0.5, 1.0, 2.0])
        result = phi_functions(z, 1)
        expected = (np.exp(z) - 1) / z
        np.testing.assert_array_almost_equal(result[..., 1], expected, decimal=10)

    def test_phi1_small_z(self):
        """Test phi_1 for small z (Taylor series path)."""
        z = np.array([1e-6, 1e-8])
        result = phi_functions(z, 1)
        # phi_1(0) = 1
        np.testing.assert_array_almost_equal(result[..., 1], np.ones(2), decimal=5)

    def test_phi2_formula(self):
        """Test phi_2 = (exp(z) - 1 - z) / z^2."""
        z = np.array([0.5, 1.0, 2.0])
        result = phi_functions(z, 2)
        expected = (np.exp(z) - 1 - z) / z ** 2
        np.testing.assert_array_almost_equal(result[..., 2], expected, decimal=10)

    def test_phi3_formula(self):
        """Test phi_3 = (exp(z) - 1 - z - z^2/2) / z^3."""
        z = np.array([0.5, 1.0])
        result = phi_functions(z, 3)
        expected = (np.exp(z) - 1 - z - z ** 2 / 2) / z ** 3
        np.testing.assert_array_almost_equal(result[..., 3], expected, decimal=10)


# ================================================================
#  KIOPS Tests
# ================================================================
class TestKiops:
    """Tests for the main kiops function."""

    def test_kiops_exp_only(self):
        """Test kiops computes exp(tau*A)*b correctly."""
        n = 10
        A = np.diag(-np.arange(1, n + 1, dtype=float))
        b = np.ones(n)
        tau = 0.1

        w, stats = kiops(tau, A, b.reshape(-1, 1))

        # Compare with exact exp(tau*A)*b
        expected = expm(tau * A) @ b
        np.testing.assert_array_almost_equal(w.real, expected, decimal=6)
        assert isinstance(stats, KrylovStats)

    def test_kiops_with_sparse_matrix(self):
        """Test kiops with sparse matrix input."""
        n = 20
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
        b = np.random.randn(n)
        tau = 0.05

        config = KrylovConfig(m_init=10, m_max=20, tol=1e-8)
        w, stats = kiops(tau, A, b.reshape(-1, 1), config)

        # Compare with dense computation
        expected = expm(tau * A.toarray()) @ b
        np.testing.assert_array_almost_equal(w.real, expected, decimal=5)

    def test_kiops_with_linear_operator(self):
        """Test kiops with LinearOperator input."""
        n = 10
        diag_vals = -np.arange(1, n + 1, dtype=float)

        def matvec(v):
            return diag_vals * v

        A = LinearOperator((n, n), matvec=matvec)
        b = np.ones(n)
        tau = 0.1

        w, stats = kiops(tau, A, b.reshape(-1, 1))

        # Compare with exact
        expected = np.exp(tau * diag_vals) * b
        np.testing.assert_array_almost_equal(w.real, expected, decimal=5)

    def test_kiops_with_callable(self):
        """Test kiops with callable operator."""
        n = 5
        scale = -2.0

        def A(v):
            return scale * v

        b = np.ones(n)
        tau = 0.2

        w, stats = kiops(tau, A, b.reshape(-1, 1))

        # exp(tau * scale * I) * b = exp(tau * scale) * b
        expected = np.exp(tau * scale) * b
        np.testing.assert_array_almost_equal(w.real, expected, decimal=5)

    def test_kiops_stats_returned(self):
        """Test that kiops returns valid statistics."""
        n = 5
        A = np.diag([-1.0] * n)
        b = np.ones(n)
        tau = 0.1

        w, stats = kiops(tau, A, b.reshape(-1, 1))

        assert stats.substeps >= 1
        assert stats.final_m >= 1
        assert stats.mvec_products >= 1
        assert isinstance(stats.error_estimate, float)

    def test_kiops_multiple_phi_functions(self):
        """Test kiops with multiple phi-function terms."""
        n = 5
        A = np.diag([-1.0] * n)
        b0 = np.ones(n)
        b1 = np.ones(n) * 0.5
        tau = 0.1

        B = np.column_stack([b0, b1])
        w, stats = kiops(tau, A, B)

        # Result should be phi_0(tau*A)*b0 + tau*phi_1(tau*A)*b1
        assert w.shape == (n,)
        assert np.all(np.isfinite(w))

    def test_kiops_default_config(self):
        """Test kiops with default configuration."""
        n = 5
        A = np.eye(n) * -1
        b = np.ones(n)

        # Should not raise
        w, stats = kiops(0.1, A, b.reshape(-1, 1))
        assert np.all(np.isfinite(w))

    def test_kiops_1d_input(self):
        """Test kiops accepts 1D input for B."""
        n = 5
        A = np.eye(n) * -1
        b = np.ones(n)  # 1D input

        w, stats = kiops(0.1, A, b)
        assert w.shape == (n,)


# ================================================================
#  Integration Tests
# ================================================================
class TestKrylovIntegration:
    """Integration tests for Krylov methods."""

    def test_heat_equation_solution(self):
        """Test Krylov methods on heat equation discretization."""
        # 1D heat equation: u_t = u_xx
        n = 50
        dx = 1.0 / (n + 1)

        # Finite difference Laplacian (with Dirichlet BCs)
        A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n), dtype=float) / dx ** 2

        # Initial condition: sin(pi*x)
        x = np.linspace(dx, 1 - dx, n)
        u0 = np.sin(np.pi * x)

        tau = 0.001  # Small time step

        w, stats = kiops(tau, A, u0.reshape(-1, 1))

        # Solution should decay (heat dissipates)
        assert np.linalg.norm(w.real) < np.linalg.norm(u0)
        assert stats.substeps >= 1

    def test_advection_equation(self):
        """Test Krylov methods on advection equation discretization."""
        # 1D advection: u_t + c*u_x = 0 with periodic BCs
        n = 32
        c = 1.0
        dx = 2 * np.pi / n

        # Upwind scheme (sparse)
        A = c * diags([-1, 1], [0, 1], shape=(n, n), dtype=float) / dx
        # Periodic BC
        A = A.tolil()
        A[n - 1, 0] = c / dx
        A = A.tocsr()

        u0 = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
        tau = 0.1

        w, stats = kiops(tau, A, u0.reshape(-1, 1))
        assert np.all(np.isfinite(w))
