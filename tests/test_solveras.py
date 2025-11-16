import pytest
import numpy as np

from rkstiff.solveras import SolverConfig, BaseSolverAS
from rkstiff.util.solver_type import SolverType


# ======================================================================
# Helper: Minimal concrete adaptive solver for testing
# ======================================================================
class MockAS(BaseSolverAS):
    """
    Minimal adaptive solver with:
    - linear operator L = diag or matrix
    - nl_func(u) = u (identity nonlinear term)
    - constant order q = 3
    - controllable update and error behavior
    """

    def __init__(self, lin_op, nl_func=None, config=SolverConfig()):
        if nl_func is None:
            nl_func = lambda u: u
        super().__init__(lin_op, nl_func, config)
        self._reset_called = False
        self.mock_unew = None
        self.mock_err = None
        self.q_value = 3

    def _reset(self):
        self._reset_called = True

    def _update_stages(self, u, h):
        # Return mock values previously set by the test
        return self.mock_unew, self.mock_err

    def _q(self):
        return self.q_value


# ======================================================================
# SolverConfig validation tests
# ======================================================================
def test_solverconfig_valid_defaults():
    cfg = SolverConfig()
    assert cfg.epsilon == 1e-4
    assert cfg.incr_f == 1.25
    assert cfg.decr_f == 0.85
    assert cfg.safety_f == 0.8
    assert cfg.adapt_cutoff == 0.01
    assert cfg.minh == 1e-16


@pytest.mark.parametrize("val", [0.0, -1e-3])
def test_solverconfig_bad_epsilon(val):
    with pytest.raises(ValueError):
        SolverConfig(epsilon=val)


@pytest.mark.parametrize("val", [1.0, 0.5, -2.0])
def test_solverconfig_bad_incr(val):
    if val <= 1.0:
        with pytest.raises(ValueError):
            SolverConfig(incr_f=val)


@pytest.mark.parametrize("val", [1.0, 2.0])
def test_solverconfig_bad_decr(val):
    with pytest.raises(ValueError):
        SolverConfig(decr_f=val)


def test_solverconfig_bad_safety():
    with pytest.raises(ValueError):
        SolverConfig(safety_f=1.01)


def test_solverconfig_bad_adapt_cutoff():
    with pytest.raises(ValueError):
        SolverConfig(adapt_cutoff=1.0)


@pytest.mark.parametrize("val", [0, -1e-5])
def test_solverconfig_bad_minh(val):
    with pytest.raises(ValueError):
        SolverConfig(minh=val)


# ======================================================================
# BaseSolverAS core behavior
# ======================================================================
def test_solver_type():
    solver = MockAS(lin_op=np.eye(2))
    assert solver.solver_type == SolverType.ADAPTIVE_STEP


def test_reset_calls_subclass_reset():
    solver = MockAS(np.eye(1))
    solver.reset()
    assert solver._reset_called is True
    assert solver.t == []
    assert solver.u == []


# ======================================================================
# Tests for step() acceptance / rejection logic
# ======================================================================
def test_step_accepts_when_s_gt_1():
    solver = MockAS(np.eye(1))

    # Mock values: small err => s >> 1
    solver.mock_unew = np.array([2.0])
    solver.mock_err = np.array([1e-12])

    u0 = np.array([1.0])
    unew, h_used, h_next = solver.step(u0, h_suggest=0.1)

    assert np.allclose(unew, solver.mock_unew)
    assert h_used == 0.1
    assert h_next > 0.1  # increased step size
    assert solver._accept is True


def test_step_rejects_when_s_lt_1():
    cfg = SolverConfig(epsilon=1e-4)
    solver = MockAS(np.eye(1), config=cfg)

    # Make error large so that s < 1
    solver.mock_unew = np.array([1.0])
    solver.mock_err = np.array([1.0])  # large err

    u0 = np.array([1.0])
    h0 = 0.1
    with pytest.raises(BaseSolverAS.MaxLoopsExceeded):
        # force infinite rejection by setting MIN_S extremely small
        solver.MAX_LOOPS = 1  # force fast fail
        solver.step(u0, h_suggest=h0)


def test_step_minimum_step_reached():
    cfg = SolverConfig(minh=1e-6)
    solver = MockAS(np.eye(1), config=cfg)

    solver.mock_unew = np.array([1.0])
    solver.mock_err = np.array([1.0])  # big error → rejection

    solver.MAX_LOOPS = 100  # allow enough loops

    # Force rejection until h < minh
    with pytest.raises(BaseSolverAS.MinimumStepReached):
        solver.step(np.array([1.0]), h_suggest=1e-5)


# ======================================================================
# Internal method tests
# ======================================================================
def test_compute_s_basic():
    cfg = SolverConfig(epsilon=1e-4, safety_f=1.0, adapt_cutoff=0.0)
    solver = MockAS(np.eye(2), config=cfg)
    solver.q_value = 1

    u = np.array([2.0, 1.0])
    err = np.array([1e-5, 1e-5])
    s = solver._compute_s(u, err)

    # Should be > 1 since err is much smaller than tolerance*||u||
    assert s > 1.0


def test_reject_step_size_nan_inf():
    solver = MockAS(np.eye(1))

    # For NaN → MIN_S * h
    h_new = solver._reject_step_size(np.nan, h=0.1)
    assert np.isclose(h_new, solver.MIN_S * 0.1)

    # For inf → MIN_S * h
    h_new = solver._reject_step_size(np.inf, h=0.1)
    assert np.isclose(h_new, solver.MIN_S * 0.1)


def test_reject_step_size_clamping():
    solver = MockAS(np.eye(1))
    solver.config.decr_f = 0.5
    solver.MIN_S = 0.2

    # s > decr_f → clamp to decr_f
    h_new = solver._reject_step_size(s=0.9, h=1.0)
    assert np.isclose(h_new, 0.5 * 1.0)


def test_accept_step_size_increase():
    solver = MockAS(np.eye(1))
    solver.MAX_S = 10.0
    solver.config.incr_f = 1.25

    s = 2.0  # > incr_f → increase step
    new = solver._accept_step_size(s, h=0.1)
    assert new > 0.1
    assert solver._accept is True


def test_accept_step_size_no_increase():
    solver = MockAS(np.eye(1))
    solver.config.incr_f = 5.0  # raise threshold

    s = 2.0  # not > incr_f → keep same h
    new = solver._accept_step_size(s, h=0.1)
    assert new == 0.1
    assert solver._accept is True


# ======================================================================
# evolve() tests
# ======================================================================
def test_evolve_runs_and_stores():
    solver = MockAS(np.eye(1))
    solver.mock_unew = np.array([1.0])
    solver.mock_err = np.array([1e-12])  # always accept

    u0 = np.array([1.0])
    u_final = solver.evolve(u0, t0=0.0, tf=0.5, h_init=0.1)

    assert np.allclose(u_final, solver.mock_unew)
    assert solver.t[0] == 0.0
    assert solver.t[-1] == pytest.approx(0.5, rel=1e-6)
    assert len(solver.u) > 1
    assert solver._reset_called is True
