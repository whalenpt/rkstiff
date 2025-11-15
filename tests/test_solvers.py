"""Unit tests for the solvers module."""

import pytest
import numpy as np
from rkstiff.etd34 import ETD34
from rkstiff.if34 import IF34
from rkstiff.if45dp import IF45DP
from rkstiff.etd import ETDAS
from rkstiff.if4 import IF4
from rkstiff.solveras import BaseSolverAS, SolverConfig
from rkstiff.etd import ETDConfig
from testing_util import (
    allen_cahn_setup,
    burgers_setup,
    kdv_soliton_setup,
    kdv_evolve_eval,
    kdv_step_eval
)


def test_bad_solver_input():
    """Test input validation for solvers."""
    # Test that invalid linear operator raises ValueError
    with pytest.raises(ValueError):
        linear_op = np.zeros(shape=(4, 2))
        ETD34(lin_op=linear_op, nl_func=None)


def test_bad_solver_config():
    """Test input validation for SolverConfig."""
    # Test that invalid linear increment factor raises ValueError
    with pytest.raises(ValueError):
        SolverConfig(incr_f=0.8)
    # Test that invalid linear decrement factor raises ValueError
    with pytest.raises(ValueError):
        SolverConfig(decr_f=1.2)
    # Test that negative epsilon raises ValueError
    with pytest.raises(ValueError):
        SolverConfig(epsilon=-1e-6)
    # Test that invalid safety_factor raises ValueError
    with pytest.raises(ValueError):
        SolverConfig(safety_f=1.2)
    # Test that invalid adapt_cutoff raises ValueError
    with pytest.raises(ValueError):
        SolverConfig(adapt_cutoff=1.0)
    with pytest.raises(ValueError):
        SolverConfig(minh=0.0)


def test_etdsolver_input():
    """Test input validation for ETD solvers."""
    # Test that modecutoff is valid
    with pytest.raises(ValueError):
        ETDConfig(modecutoff=1.2)
    with pytest.raises(ValueError):
        ETDConfig(modecutoff=0)
    # Test that contour_points is an integer
    with pytest.raises(TypeError):
        ETDConfig(contour_points=12.2)
    # Test that contour_points is greater than 1
    with pytest.raises(ValueError):
        ETDConfig(contour_points=1)
    # Test that contour_radius is positive
    with pytest.raises(ValueError):
        ETDConfig(contour_radius=0)


def test_abc_error():
    """Test that abstract base classes cannot be instantiated."""
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        ETDAS(lin_op=None, nl_func=None)
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        BaseSolverAS(lin_op=None, nl_func=None)



def test_if45dp():
    """Test the IF45DP solver on the Burgers equation."""
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF45DP(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2

