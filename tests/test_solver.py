"""Comprehensive tests for the BaseSolver abstract base class."""

import logging
import numpy as np
import pytest
from abc import ABC
from rkstiff.solver import BaseSolver
from rkstiff.util.solver_type import SolverType


# ============================================================================
# Concrete Test Implementations
# ============================================================================


class DummySolver(BaseSolver):
    """Minimal concrete implementation for testing BaseSolver."""

    @property
    def solver_type(self) -> SolverType:
        """Return CONSTANT_STEP for testing."""
        return SolverType.CONSTANT_STEP

    def reset(self) -> None:
        """Reset solver state."""
        self.t, self.u = [], []
        self._reset()

    def _reset(self) -> None:
        """Reset internal state (placeholder for testing)."""
        pass


class IncompleteSolver(BaseSolver):
    """Solver missing required abstract methods - should not instantiate."""

    pass


# ============================================================================
# Initialization Tests
# ============================================================================


def test_base_solver_diagonal_operator():
    """Test initialization with 1D diagonal operator."""
    lin_op = np.array([1.0, 2.0, 3.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Check attributes
    assert np.array_equal(solver.lin_op, lin_op)
    assert solver.nl_func is nl_func
    assert solver._diag is True
    assert isinstance(solver.t, list)
    assert isinstance(solver.u, list)
    assert len(solver.t) == 0
    assert len(solver.u) == 0


def test_base_solver_matrix_operator():
    """Test initialization with 2D square matrix operator."""
    lin_op = np.array([[1.0, 2.0], [3.0, 4.0]])
    nl_func = lambda u: -(u**3)

    solver = DummySolver(lin_op, nl_func)

    # Check attributes
    assert np.array_equal(solver.lin_op, lin_op)
    assert solver.nl_func is nl_func
    assert solver._diag is False


def test_base_solver_large_diagonal():
    """Test with large diagonal operator."""
    n = 1000
    lin_op = np.linspace(-10, 10, n)
    nl_func = lambda u: np.sin(u)

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is True
    assert solver.lin_op.shape == (n,)


def test_base_solver_large_matrix():
    """Test with large matrix operator."""
    n = 100
    lin_op = np.random.randn(n, n)
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is False
    assert solver.lin_op.shape == (n, n)


# ============================================================================
# Validation Tests
# ============================================================================


def test_base_solver_invalid_1d_shape():
    """Test that 3D arrays are rejected."""
    lin_op = np.ones((3, 3, 3))
    nl_func = lambda u: u

    with pytest.raises(ValueError, match="lin_op must be 1D or 2D"):
        DummySolver(lin_op, nl_func)


def test_base_solver_invalid_0d_shape():
    """Test that scalar values are rejected."""
    lin_op = np.array(5.0)  # 0D array
    nl_func = lambda u: u

    with pytest.raises(ValueError, match="lin_op must be 1D or 2D"):
        DummySolver(lin_op, nl_func)


def test_base_solver_non_square_matrix():
    """Test that non-square matrices are rejected."""
    lin_op = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3 matrix
    nl_func = lambda u: u**2

    with pytest.raises(ValueError, match="lin_op must be a square matrix"):
        DummySolver(lin_op, nl_func)


def test_base_solver_rectangular_matrix():
    """Test rejection of tall rectangular matrix."""
    lin_op = np.random.randn(5, 3)
    nl_func = lambda u: u

    with pytest.raises(ValueError, match="lin_op must be a square matrix"):
        DummySolver(lin_op, nl_func)


# ============================================================================
# Logging Tests
# ============================================================================


def test_base_solver_default_loglevel():
    """Test that default log level is WARNING."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver.logger.level == logging.WARNING


def test_base_solver_custom_loglevel_string():
    """Test initialization with custom log level (string)."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        solver = DummySolver(lin_op, nl_func, loglevel=level)
        assert solver.logger.level == getattr(logging, level)


def test_base_solver_custom_loglevel_int():
    """Test initialization with custom log level (integer)."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func, loglevel=logging.DEBUG)
    assert solver.logger.level == logging.DEBUG

    solver = DummySolver(lin_op, nl_func, loglevel=logging.ERROR)
    assert solver.logger.level == logging.ERROR


def test_base_solver_set_loglevel_string():
    """Test changing log level at runtime with string."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func, loglevel="WARNING")
    assert solver.logger.level == logging.WARNING

    solver.set_loglevel("DEBUG")
    assert solver.logger.level == logging.DEBUG

    solver.set_loglevel("ERROR")
    assert solver.logger.level == logging.ERROR


def test_base_solver_set_loglevel_int():
    """Test changing log level at runtime with integer."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    solver.set_loglevel(logging.INFO)
    assert solver.logger.level == logging.INFO

    solver.set_loglevel(logging.CRITICAL)
    assert solver.logger.level == logging.CRITICAL


def test_base_solver_logger_name():
    """Test that logger has correct name."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Logger name should contain the class name
    assert "DummySolver" in solver.logger.name or "rkstiff" in solver.logger.name


def test_base_solver_logging_output(caplog):
    """Test that logging produces expected messages."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    with caplog.at_level(logging.INFO):
        solver = DummySolver(lin_op, nl_func, loglevel="INFO")

    # Should log initialization message
    assert any("Initialized" in record.message for record in caplog.records)


def test_base_solver_debug_logging(caplog):
    """Test debug-level logging messages."""
    lin_op = np.array([1.0, 2.0, 3.0])
    nl_func = lambda u: u**2

    with caplog.at_level(logging.DEBUG):
        solver = DummySolver(lin_op, nl_func, loglevel="DEBUG")

    # Should log shape and diagonal information
    assert any("shape" in record.message.lower() for record in caplog.records)
    assert any("diagonal" in record.message.lower() for record in caplog.records)


# ============================================================================
# Abstract Method Tests
# ============================================================================


def test_base_solver_cannot_instantiate_incomplete():
    """Test that BaseSolver with missing abstract methods cannot be instantiated."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteSolver(lin_op, nl_func)


def test_base_solver_abstract_methods_required():
    """Test that all abstract methods must be implemented."""

    # Missing both reset and _reset
    class MissingBoth(BaseSolver):
        @property
        def solver_type(self):
            return SolverType.CONSTANT_STEP

    # Missing solver_type property
    class MissingSolverType(BaseSolver):
        def reset(self):
            pass

        def _reset(self):
            pass

    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    with pytest.raises(TypeError):
        MissingBoth(lin_op, nl_func)

    with pytest.raises(TypeError):
        MissingSolverType(lin_op, nl_func)


# ============================================================================
# Solver Type Property Tests
# ============================================================================


def test_base_solver_solver_type_property():
    """Test that solver_type property works correctly."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver.solver_type == SolverType.CONSTANT_STEP
    assert isinstance(solver.solver_type, SolverType)


def test_base_solver_solver_type_is_abstract():
    """Test that solver_type must be implemented by subclasses."""

    class NoSolverType(BaseSolver):
        def reset(self):
            pass

        def _reset(self):
            pass

    lin_op = np.array([1.0])
    nl_func = lambda u: u

    with pytest.raises(TypeError, match="abstract"):
        NoSolverType(lin_op, nl_func)


# ============================================================================
# Reset Tests
# ============================================================================


def test_base_solver_reset_clears_data():
    """Test that reset clears time and solution arrays."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Add some data
    solver.t = [0.0, 1.0, 2.0]
    solver.u = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

    # Reset
    solver.reset()

    # Data should be cleared
    assert len(solver.t) == 0
    assert len(solver.u) == 0


def test_base_solver_reset_calls_internal_reset():
    """Test that reset() calls _reset()."""

    class TrackedSolver(BaseSolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reset_called = False

        @property
        def solver_type(self):
            return SolverType.CONSTANT_STEP

        def reset(self):
            self.t, self.u = [], []
            self._reset()

        def _reset(self):
            self._reset_called = True

    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = TrackedSolver(lin_op, nl_func)
    assert not solver._reset_called

    solver.reset()
    assert solver._reset_called


def test_base_solver_multiple_resets():
    """Test that reset can be called multiple times."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Reset multiple times
    for _ in range(5):
        solver.t = [0.0, 1.0]
        solver.u = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        solver.reset()
        assert len(solver.t) == 0
        assert len(solver.u) == 0


# ============================================================================
# Nonlinear Function Tests
# ============================================================================


def test_base_solver_nonlinear_function_call():
    """Test that nonlinear function can be called."""
    lin_op = np.array([1.0, 2.0, 3.0])
    nl_func = lambda u: u**2 + u

    solver = DummySolver(lin_op, nl_func)

    u = np.array([1.0, 2.0, 3.0])
    result = solver.nl_func(u)
    expected = u**2 + u

    assert np.allclose(result, expected)


def test_base_solver_complex_nonlinear_function():
    """Test with complex-valued nonlinear function."""
    lin_op = np.array([1.0j, -2.0j])
    nl_func = lambda u: -1j * np.abs(u) ** 2 * u

    solver = DummySolver(lin_op, nl_func)

    u = np.array([1.0 + 1.0j, 2.0 - 0.5j])
    result = solver.nl_func(u)

    assert result.dtype == np.complex128
    assert result.shape == u.shape


def test_base_solver_stateful_nonlinear_function():
    """Test with a stateful/closure nonlinear function."""
    lin_op = np.array([1.0, 2.0])

    # Nonlinear function that depends on external parameter
    alpha = 0.5
    nl_func = lambda u: alpha * u**2

    solver = DummySolver(lin_op, nl_func)

    u = np.array([2.0, 4.0])
    result = solver.nl_func(u)
    expected = alpha * u**2

    assert np.allclose(result, expected)


# ============================================================================
# Edge Cases
# ============================================================================


def test_base_solver_single_element_diagonal():
    """Test with single-element diagonal operator."""
    lin_op = np.array([5.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is True
    assert solver.lin_op.shape == (1,)


def test_base_solver_1x1_matrix():
    """Test with 1x1 matrix operator."""
    lin_op = np.array([[5.0]])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is False
    assert solver.lin_op.shape == (1, 1)


def test_base_solver_zero_operator():
    """Test with zero operator."""
    lin_op = np.zeros(5)
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is True
    assert np.all(solver.lin_op == 0)


def test_base_solver_identity_matrix():
    """Test with identity matrix operator."""
    n = 10
    lin_op = np.eye(n)
    nl_func = lambda u: u

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is False
    assert np.allclose(solver.lin_op, np.eye(n))


def test_base_solver_complex_operator():
    """Test with complex-valued operator."""
    lin_op = np.array([1.0j, -2.0j, 3.0 + 1.0j])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is True
    assert solver.lin_op.dtype == np.complex128


def test_base_solver_sparse_like_matrix():
    """Test with mostly-zero matrix (sparse-like structure)."""
    n = 20
    lin_op = np.zeros((n, n))
    lin_op[0, 0] = 1.0
    lin_op[-1, -1] = 2.0
    nl_func = lambda u: u

    solver = DummySolver(lin_op, nl_func)

    assert solver._diag is False
    assert solver.lin_op.shape == (n, n)


# ============================================================================
# Data Storage Tests
# ============================================================================


def test_base_solver_time_storage():
    """Test that time data can be stored and accessed."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Manually add time points
    times = [0.0, 0.1, 0.2, 0.3]
    solver.t = times.copy()

    assert len(solver.t) == len(times)
    assert solver.t == times


def test_base_solver_solution_storage():
    """Test that solution data can be stored and accessed."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Manually add solutions
    solutions = [np.array([1.0, 2.0]), np.array([1.1, 2.1]), np.array([1.2, 2.2])]
    solver.u = solutions

    assert len(solver.u) == len(solutions)
    for i, sol in enumerate(solutions):
        assert np.array_equal(solver.u[i], sol)


def test_base_solver_empty_storage_initialization():
    """Test that storage is initialized empty."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    assert isinstance(solver.t, list)
    assert isinstance(solver.u, list)
    assert len(solver.t) == 0
    assert len(solver.u) == 0


# ============================================================================
# Type Checking and Attributes
# ============================================================================


def test_base_solver_is_abstract():
    """Test that BaseSolver is an abstract base class."""
    assert issubclass(BaseSolver, ABC)
    assert hasattr(BaseSolver, "__abstractmethods__")


def test_base_solver_has_required_attributes():
    """Test that solver has all expected attributes."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Check all expected attributes exist
    assert hasattr(solver, "lin_op")
    assert hasattr(solver, "nl_func")
    assert hasattr(solver, "logger")
    assert hasattr(solver, "t")
    assert hasattr(solver, "u")
    assert hasattr(solver, "_diag")
    assert hasattr(solver, "solver_type")


def test_base_solver_has_required_methods():
    """Test that solver has all expected methods."""
    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    solver = DummySolver(lin_op, nl_func)

    # Check all expected methods exist and are callable
    assert hasattr(solver, "reset") and callable(solver.reset)
    assert hasattr(solver, "_reset") and callable(solver._reset)
    assert hasattr(solver, "set_loglevel") and callable(solver.set_loglevel)


def test_base_solver_diagonal_detection_1d():
    """Test that 1D arrays are correctly identified as diagonal."""
    test_cases = [
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array(range(100)),
        np.linspace(-10, 10, 50),
    ]

    nl_func = lambda u: u**2

    for lin_op in test_cases:
        solver = DummySolver(lin_op, nl_func)
        assert solver._diag is True, f"Failed for shape {lin_op.shape}"


def test_base_solver_diagonal_detection_2d():
    """Test that 2D arrays are correctly identified as non-diagonal."""
    test_cases = [
        np.array([[1.0]]),
        np.eye(2),
        np.eye(10),
        np.random.randn(5, 5),
        np.zeros((3, 3)),
    ]

    nl_func = lambda u: u**2

    for lin_op in test_cases:
        solver = DummySolver(lin_op, nl_func)
        assert solver._diag is False, f"Failed for shape {lin_op.shape}"


# ============================================================================
# Integration with Concrete Solvers
# ============================================================================


def test_base_solver_integration_with_if4():
    """Test that BaseSolver works with actual IF4 solver."""
    from rkstiff.if4 import IF4

    lin_op = np.array([1.0, -2.0, 3.0])
    nl_func = lambda u: u**2

    solver = IF4(lin_op, nl_func)

    # Should have all BaseSolver attributes
    assert hasattr(solver, "lin_op")
    assert hasattr(solver, "nl_func")
    assert hasattr(solver, "logger")
    assert hasattr(solver, "t")
    assert hasattr(solver, "u")
    assert hasattr(solver, "_diag")

    # Should have solver_type property
    assert solver.solver_type == SolverType.CONSTANT_STEP


# ============================================================================
# Inheritance Tests
# ============================================================================


def test_base_solver_multiple_subclasses():
    """Test that multiple solvers can be created from BaseSolver."""

    class Solver1(BaseSolver):
        @property
        def solver_type(self):
            return SolverType.CONSTANT_STEP

        def reset(self):
            self.t, self.u = [], []
            self._reset()

        def _reset(self):
            pass

    class Solver2(BaseSolver):
        @property
        def solver_type(self):
            return SolverType.ADAPTIVE_STEP

        def reset(self):
            self.t, self.u = [], []
            self._reset()

        def _reset(self):
            pass

    lin_op = np.array([1.0, 2.0])
    nl_func = lambda u: u**2

    s1 = Solver1(lin_op, nl_func)
    s2 = Solver2(lin_op, nl_func)

    assert s1.solver_type == SolverType.CONSTANT_STEP
    assert s2.solver_type == SolverType.ADAPTIVE_STEP

    # Should be independent instances
    s1.t = [1.0]
    assert len(s2.t) == 0
