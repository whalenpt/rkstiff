import pytest
import numpy as np

from rkstiff.solvercs import BaseSolverCS
from rkstiff.util.solver_type import SolverType


# ======================================================================
# Minimal concrete solver for testing BaseSolverCS
# ======================================================================
class MockCS(BaseSolverCS):
    def __init__(self, lin_op, nl_func=None, **kwargs):
        if nl_func is None:
            nl_func = lambda u: u
        super().__init__(lin_op, nl_func, **kwargs)
        self._reset_called = False
        self.mock_unew = None
        self.last_u = None
        self.last_h = None

    def _reset(self):
        self._reset_called = True

    def _update_stages(self, u, h):
        # record calls for testing
        self.last_u = u
        self.last_h = h
        return self.mock_unew


# ======================================================================
# Tests: solver_type
# ======================================================================
def test_solver_type_is_constant_step():
    solver = MockCS(np.eye(2))
    assert solver.solver_type == SolverType.CONSTANT_STEP


# ======================================================================
# Tests: reset()
# ======================================================================
def test_reset_clears_state_and_calls_subclass_reset():
    solver = MockCS(np.eye(1))
    solver.t = [1.0, 2.0]
    solver.u = [np.array([1.0])]
    solver.reset()

    assert solver.t == []
    assert solver.u == []
    assert solver._reset_called is True


# ======================================================================
# Tests: step()
# ======================================================================
def test_step_calls_update_stages_and_returns_value():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([2.0])

    u0 = np.array([1.0])
    out = solver.step(u0, h=0.1)

    assert np.allclose(out, solver.mock_unew)
    assert np.allclose(solver.last_u, u0)
    assert solver.last_h == 0.1


def test_step_asserts_nonnegative_h():
    solver = MockCS(np.eye(1))
    with pytest.raises(AssertionError):
        solver.step(np.array([1.0]), h=-0.1)


# ======================================================================
# Tests: evolve() core behavior
# ======================================================================
def test_evolve_runs_and_stores_snapshots():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([10.0])  # return same each step

    u0 = np.array([1.0])
    u_final = solver.evolve(u0, t0=0.0, tf=0.3, h=0.1)

    assert solver._reset_called is True

    # Expected time points: [0.0, 0.1, 0.2, 0.3]
    assert len(solver.t) == 4
    assert solver.t[0] == 0.0
    assert solver.t[-1] == pytest.approx(0.3)

    assert np.allclose(u_final, solver.mock_unew)
    assert len(solver.u) == 4


def test_evolve_no_store_data():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([5.0])

    u0 = np.array([1.0])
    u_final = solver.evolve(u0, t0=0.0, tf=0.2, h=0.1, store_data=False)

    assert solver.t == []  # nothing stored
    assert solver.u == []
    assert np.allclose(u_final, solver.mock_unew)


def test_evolve_store_every_other_step():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([3.0])

    u0 = np.array([1.0])
    solver.evolve(u0, t0=0.0, tf=0.4, h=0.1, store_data=True, store_freq=2)

    # Steps happen at: 0.1, 0.2, 0.3, 0.4
    # Store at steps 2 and 4 → times 0.2 and 0.4, plus initial t0
    assert solver.t == [0.0, 0.2, 0.4]


# ======================================================================
# Tests: evolve() error conditions
# ======================================================================
def test_evolve_raises_if_h_too_large():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([2.0])

    u0 = np.array([1.0])

    with pytest.raises(ValueError):
        solver.evolve(u0, t0=0.0, tf=0.1, h=0.2)  # h > (tf - t0)


def test_evolve_calls_step_with_correct_h():
    solver = MockCS(np.eye(1))
    solver.mock_unew = np.array([7.0])

    u0 = np.array([1.0])
    solver.evolve(u0, t0=0.0, tf=0.3, h=0.1)

    # Should have 3 calls to step with h = 0.1
    assert solver.last_h == 0.1


def test_evolve_final_solution_is_correct():
    solver = MockCS(np.eye(1))

    # each step multiplies by 2
    def mult2(u): return u * 2
    solver.mock_unew = None  # not used for logical update

    # Override update_stages to apply operator
    def _update(u, h):
        return mult2(u)

    solver._update_stages = _update

    u0 = np.array([1.0])
    u_final = solver.evolve(u0, t0=0.0, tf=0.3, h=0.1)

    # 3 steps: 1 → 2 → 4 → 8
    assert np.allclose(u_final, np.array([8.0]))
