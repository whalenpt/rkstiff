"""Unit tests for the solvers module."""

import pytest
import numpy as np
from rkstiff.grids import construct_x_kx_rfft, construct_x_dx_cheb
from rkstiff.etd34 import ETD34
from rkstiff.etd35 import ETD35
from rkstiff.if34 import IF34
from rkstiff.if45dp import IF45DP
from rkstiff.etd import ETDAS
from rkstiff.etd4 import ETD4
from rkstiff.etd5 import ETD5
from rkstiff.if4 import IF4
from rkstiff.solver import StiffSolverAS, SolverConfig
from rkstiff.etd import ETDConfig
from rkstiff import models


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
    # Test that invalid safetyFactor raises ValueError
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
        StiffSolverAS(lin_op=None, nl_func=None)


def allen_cahn_setup():
    """Set up the Allen-Cahn problem for testing solvers."""
    n = 20
    a = -1
    b = 1
    x, d_cheb_matrix = construct_x_dx_cheb(n, a, b)
    epsilon = 0.01
    linear_op, nl_func = models.allen_cahn_ops(x, d_cheb_matrix, epsilon)
    u0 = 0.53 * x + 0.47 * np.sin(-1.5 * np.pi * x)
    w0 = u0 - x
    w0int = w0[1:-1]
    u0int = u0[1:-1]
    xint = x[1:-1]
    return xint, u0int, w0int, linear_op, nl_func


def burgers_setup():
    """Set up the Burgers problem for testing solvers."""
    n = 1024
    a, b = -np.pi, np.pi
    x, kx = construct_x_kx_rfft(n, a, b)

    mu = 0.0005
    linear_op, nl_func = models.burgers_ops(kx, mu)

    u0 = np.exp(-10 * np.sin(x / 2) ** 2)
    u0_fft = np.fft.rfft(u0)
    return u0_fft, linear_op, nl_func


def kdv_soliton_setup():
    """Set up the KdV soliton problem for testing solvers."""
    x, kx = construct_x_kx_rfft(256, -30, 30)
    ampl, x0, t0 = 1.0, -5.0, 0

    h = 0.025
    steps = 200
    tf = h * steps

    u0 = models.kdv_soliton(x, ampl=ampl, x0=x0, t=t0)
    u0_fft = np.fft.rfft(u0)
    u_exact_fft = np.fft.rfft(models.kdv_soliton(x, ampl=ampl, x0=x0, t=tf))
    linear_op, nl_func = models.kdv_ops(kx)

    return u0_fft, linear_op, nl_func, u_exact_fft, h, steps


def kdv_cs_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol):
    """Test the step method of a solver on the KdV soliton problem."""
    for _ in range(steps):
        u0_fft = solver.step(u0_fft, h)
    rel_err = np.linalg.norm(u0_fft - u_exact_fft) / np.linalg.norm(u_exact_fft)
    assert rel_err < tol


def kdv_adp_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol):
    """Test the step method of a solver on the KdV soliton problem."""
    for _ in range(steps):
        u0_fft, h_actual, _ = solver.step(u0_fft, h)
        assert (h_actual - h) < 1e-10
    rel_err = np.linalg.norm(u0_fft - u_exact_fft) / np.linalg.norm(u_exact_fft)
    assert rel_err < tol


def kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf, tol):
    """Test the evolve method of a solver on the KdV soliton problem."""
    u_fft = solver.evolve(u0_fft, 0.0, tf, h, store_data=False)
    rel_err = np.linalg.norm(u_fft - u_exact_fft) / np.linalg.norm(u_exact_fft)
    assert rel_err < tol


def test_etd34_nondiag():
    """Test the ETD34 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    config = SolverConfig(epsilon=1e-3)
    etd_config = ETDConfig(contour_points=64, contour_radius=20)
    solver = ETD34(lin_op=linear_op, nl_func=nl_func, config=config, etd_config=etd_config)
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_etd34():
    """Test the ETD34 solver on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD34(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1)
    )  # small epsilon -> step size isn't auto-reduced
    kdv_adp_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)
    solver.reset()
    solver.config.epsilon = 1e-4
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd35():
    """Test the ETD35 solver on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD34(
        lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-1)
    )  # small epsilon -> step size isn't auto-reduced
    kdv_adp_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-4)
    solver.reset()
    solver.config.epsilon = 1e-4
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-4)


def test_etd35_nondiag():
    """Test the ETD35 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    config = SolverConfig(epsilon=1e-4)
    etd_config = ETDConfig(contour_points=32, contour_radius=10)
    solver = ETD35(lin_op=linear_op, nl_func=nl_func, config=config, etd_config=etd_config)
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_if34():
    """Test the IF34 solver on the Burgers equation."""
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func)
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2


def test_if34_nondiag():
    """Test the IF34 solver on the Allen-Cahn equation."""
    xint, u0int, w0int, linear_op, nl_func = allen_cahn_setup()
    solver = IF34(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    wfint = solver.evolve(w0int, t0=0, tf=60, store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0] - ufint[0]) < 0.01
    assert np.abs(u0int[7] - ufint[7]) > 1


def test_if45dp():
    """Test the IF45DP solver on the Burgers equation."""
    u0_fft, linear_op, nl_func = burgers_setup()
    solver = IF45DP(lin_op=linear_op, nl_func=nl_func, config=SolverConfig(epsilon=1e-3))
    u_fft = solver.evolve(u0_fft, t0=0, tf=0.85, store_data=False)
    rel_err = np.abs(np.linalg.norm(u_fft) - np.linalg.norm(u0_fft)) / np.linalg.norm(u0_fft)
    assert rel_err < 1e-2


def test_etd4_step():
    """Test the ETD4 solver step method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD4(lin_op=linear_op, nl_func=nl_func)
    kdv_cs_step_eval(solver, u0_fft, u_exact_fft, h, steps, 1e-6)


def test_etd4_evolve():
    """Test the ETD4 solver evolve method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD4(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf=h * steps, tol=1e-6)


def test_etd5_step():
    """Test the ETD5 solver step method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD5(lin_op=linear_op, nl_func=nl_func)
    kdv_cs_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-6)


def test_etd5_evolve():
    """Test the ETD5 solver evolve method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = ETD5(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h=h, tf=h * steps, tol=1e-6)


def test_if4_step():
    """Test the IF4 solver step method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF4(lin_op=linear_op, nl_func=nl_func)
    kdv_cs_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol=1e-5)


def test_if4_evolve():
    """Test the IF4 solver evolve method on the KdV soliton problem."""
    u0_fft, linear_op, nl_func, u_exact_fft, h, steps = kdv_soliton_setup()
    solver = IF4(lin_op=linear_op, nl_func=nl_func)
    kdv_evolve_eval(solver, u0_fft, u_exact_fft, h=h, tf=h * steps, tol=1e-5)
