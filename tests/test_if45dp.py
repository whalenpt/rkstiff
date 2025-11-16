"""Tests for the IF45 solver and its components."""

import numpy as np
import pytest
from rkstiff.if45dp import IF45DP
from rkstiff.solveras import SolverConfig
from testing_util import burgers_setup


def test_if45dp():
    """Test the IF45DP solver on the Burgers equation."""
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF45DP(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2

def test_if45dp_rejects_non_1d_operator():
    lin_op = np.eye(3)
    nl = lambda u: u
    with pytest.raises(ValueError) as e:
        IF45DP(lin_op=lin_op, nl_func=nl, config=SolverConfig())

    assert "IF45DP only handles 1D linear operators" in str(e.value)
