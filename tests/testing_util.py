"""Utility functions for setting up test problems."""

from rkstiff.grids import construct_x_kx_rfft, construct_x_dx_cheb
from rkstiff import models
import numpy as np


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


def kdv_evolve_eval(solver, u0_fft, u_exact_fft, h, tf, tol):
    """Test the evolve method of a solver on the KdV soliton problem."""
    u_fft = solver.evolve(u0_fft, 0.0, tf, h, store_data=False)
    rel_err = np.linalg.norm(u_fft - u_exact_fft) / np.linalg.norm(u_exact_fft)
    assert rel_err < tol


def kdv_step_eval(solver, u0_fft, u_exact_fft, h, steps, tol, solver_type='cs'):
    """Test the step method of a solver on the KdV soliton problem."""
    if solver_type == 'cs':
        for _ in range(steps):
            u0_fft = solver.step(u0_fft, h)
    elif solver_type == 'as':
        for _ in range(steps):
            u0_fft, h_actual, _ = solver.step(u0_fft, h)
            assert (h_actual - h) < 1e-10  # Ensure step size matches requested... so no adaptivity
    rel_err = np.linalg.norm(u0_fft - u_exact_fft) / np.linalg.norm(u_exact_fft)
    assert rel_err < tol
