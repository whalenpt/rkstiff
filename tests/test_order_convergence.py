"""Order convergence tests for ETD5 and other solvers."""

import pytest
import numpy as np
from rkstiff.etd4 import ETD4
from rkstiff.etd5 import ETD5
from rkstiff.if4 import IF4
from rkstiff.etd import ETDConfig
from testing_util import (
    allen_cahn_setup,
    burgers_setup,
    kdv_soliton_setup,
)


def compute_convergence_order(errors, step_sizes):
    """
    Compute convergence order using least squares fit in log-log space.
    This is more robust than pairwise comparisons, especially when
    step sizes don't follow a constant refinement ratio.
    """
    log_h = np.log(step_sizes)
    log_err = np.log(errors)
    
    # Fit: log(error) = order * log(h) + const
    # This gives error ~ h^order
    order = np.polyfit(log_h, log_err, 1)[0]
    
    return order


def kdv_order_test(solver, expected_order, step_sizes, tolerance=0.5):
    """
    Test convergence order of a solver on the KdV soliton problem.
    
    Parameters
    ----------
    solver : BaseSolver
        Initialized solver instance.
    expected_order : float
        Expected convergence order.
    step_sizes : list
        List of step sizes to test (coarse to fine).
    tolerance : float
        Tolerance for order check (expected_order - tolerance).
    """
    u0_fft, linear_op, nl_func, _, _, _ = kdv_soliton_setup()
    
    # Use a shorter time interval for order tests
    t_final = 1.0
    
    # Generate reference solution with very small step size
    h_ref = step_sizes[-1] / 4
    n_steps_ref = int(t_final / h_ref)
    
    # Create solver_ref with appropriate arguments
    solver_kwargs = {'lin_op': linear_op, 'nl_func': nl_func}
    if hasattr(solver, 'config'):
        solver_kwargs['config'] = solver.config
    if hasattr(solver, 'etd_config'):
        solver_kwargs['etd_config'] = solver.etd_config
    solver_ref = solver.__class__(**solver_kwargs)
    
    u_ref = u0_fft.copy()
    for _ in range(n_steps_ref):
        u_ref = solver_ref.step(u_ref, h_ref)
    
    # Compute errors at different step sizes
    errors = []
    for h in step_sizes:
        n_steps = int(t_final / h)
        test_solver = solver.__class__(**solver_kwargs)
        u = u0_fft.copy()
        for _ in range(n_steps):
            u = test_solver.step(u, h)
        
        error = np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
        errors.append(error)
    
    # Compute convergence order using least squares
    order = compute_convergence_order(errors, step_sizes)
    
    print(f"\n{solver.__class__.__name__} KdV Convergence Test:")
    print(f"Step sizes: {step_sizes}")
    print(f"Errors: {[f'{e:.2e}' for e in errors]}")
    print(f"Computed Order: {order:.3f} (expected ~{expected_order})")
    
    assert order > expected_order - tolerance, \
        f"Expected order ~{expected_order}, got {order:.3f}"
    
    return order, errors


def burgers_order_test(solver, expected_order, step_sizes, tolerance=0.5):
    """
    Test convergence order of a solver on the Burgers equation.
    
    Parameters
    ----------
    solver : BaseSolver
        Initialized solver instance.
    expected_order : float
        Expected convergence order.
    step_sizes : list
        List of step sizes to test (coarse to fine).
    tolerance : float
        Tolerance for order check (expected_order - tolerance).
    """
    u0_fft, linear_op, nl_func = burgers_setup()
    
    # Use a shorter time interval
    t_final = 0.5
    
    # Generate reference solution
    h_ref = step_sizes[-1] / 4
    n_steps_ref = int(t_final / h_ref)
    
    # Create solver_ref with appropriate arguments
    solver_kwargs = {'lin_op': linear_op, 'nl_func': nl_func}
    if hasattr(solver, 'config'):
        solver_kwargs['config'] = solver.config
    if hasattr(solver, 'etd_config'):
        solver_kwargs['etd_config'] = solver.etd_config
    solver_ref = solver.__class__(**solver_kwargs)
    
    u_ref = u0_fft.copy()
    for _ in range(n_steps_ref):
        u_ref = solver_ref.step(u_ref, h_ref)
    
    # Compute errors at different step sizes
    errors = []
    for h in step_sizes:
        n_steps = int(t_final / h)
        test_solver = solver.__class__(**solver_kwargs)
        u = u0_fft.copy()
        for _ in range(n_steps):
            u = test_solver.step(u, h)
        
        error = np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
        errors.append(error)
    
    # Compute convergence order using least squares
    order = compute_convergence_order(errors, step_sizes)
    
    print(f"\n{solver.__class__.__name__} Burgers Convergence Test:")
    print(f"Step sizes: {step_sizes}")
    print(f"Errors: {[f'{e:.2e}' for e in errors]}")
    print(f"Computed Order: {order:.3f} (expected ~{expected_order})")
    
    assert order > expected_order - tolerance, \
        f"Expected order ~{expected_order}, got {order:.3f}"
    
    return order, errors


def allen_cahn_order_test(solver, expected_order, step_sizes, tolerance=0.5):
    """
    Test convergence order of a solver on the Allen-Cahn equation.
    
    Parameters
    ----------
    solver : BaseSolver
        Initialized solver instance.
    expected_order : float
        Expected convergence order.
    step_sizes : list
        List of step sizes to test (coarse to fine).
    tolerance : float
        Tolerance for order check (expected_order - tolerance).
    """
    xint, _, w0int, linear_op, nl_func = allen_cahn_setup()
    
    # Use a short time interval
    t_final = 1.0 
    
    # Generate reference solution
    h_ref = step_sizes[-1] / 4
    n_steps_ref = int(t_final / h_ref)
    
    # Create solver_ref with appropriate arguments
    solver_kwargs = {'lin_op': linear_op, 'nl_func': nl_func}
    if hasattr(solver, 'config'):
        solver_kwargs['config'] = solver.config
    if hasattr(solver, 'etd_config'):
        solver_kwargs['etd_config'] = solver.etd_config
    solver_ref = solver.__class__(**solver_kwargs)
    
    w_ref = w0int.copy()
    for _ in range(n_steps_ref):
        w_ref = solver_ref.step(w_ref, h_ref)
    uf_ref = w_ref.real + xint
    
    # Compute errors at different step sizes
    errors = []
    for h in step_sizes:
        n_steps = int(t_final / h)
        test_solver = solver.__class__(**solver_kwargs)
        w = w0int.copy()
        for _ in range(n_steps):
            w = test_solver.step(w, h)
        uf = w.real + xint
        
        error = np.linalg.norm(uf - uf_ref) / np.linalg.norm(uf_ref)
        errors.append(error)
    
    # Compute convergence order using least squares
    order = compute_convergence_order(errors, step_sizes)
    
    print(f"\n{solver.__class__.__name__} Allen-Cahn Convergence Test:")
    print(f"Step sizes: {step_sizes}")
    print(f"Errors: {[f'{e:.2e}' for e in errors]}")
    print(f"Computed Order: {order:.3f} (expected ~{expected_order})")
    
    # Allen-Cahn is very stiff, use more tolerance
    assert order > expected_order - tolerance - 0.5, \
        f"Expected order ~{expected_order}, got {order:.3f}"
    
    return order, errors


class TestIF4Order:
    """Test convergence order of the IF4 solver."""

    def test_etd4_burgers_order(self):
        """Test ETD4 convergence order on Burgers equation (diagonal system)."""
        _, linear_op, nl_func = burgers_setup()
        solver = IF4(lin_op=linear_op, nl_func=nl_func)
        step_sizes = [0.005, 0.0025, 0.00125, 0.000625]
        order, _ = burgers_order_test(solver, expected_order=4.0,
                                             step_sizes=step_sizes, tolerance=0.5)
        assert order > 3.5

    def test_etd4_kdv_order(self):
        """Test ETD4 convergence order on KdV equation."""
        _, linear_op, nl_func, _, _, _ = kdv_soliton_setup()
        solver = ETD4(lin_op=linear_op, nl_func=nl_func)
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        order, _ = kdv_order_test(solver, expected_order=4.0,
                                         step_sizes=step_sizes, tolerance=0.5)
        assert order > 3.5


class TestETD4Order:
    """Test convergence order of the ETD4 solver."""
    
    def test_etd4_burgers_order(self):
        """Test ETD4 convergence order on Burgers equation (diagonal system)."""
        _, linear_op, nl_func = burgers_setup()
        solver = ETD4(lin_op=linear_op, nl_func=nl_func)
        step_sizes = [0.005, 0.0025, 0.00125, 0.000625]
        order, _ = burgers_order_test(solver, expected_order=4.0,
                                             step_sizes=step_sizes, tolerance=0.5)
        assert order > 3.5

    def test_etd4_kdv_order(self):
        """Test ETD4 convergence order on KdV equation."""
        _, linear_op, nl_func, _, _, _ = kdv_soliton_setup()
        solver = ETD4(lin_op=linear_op, nl_func=nl_func)
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        order, _ = kdv_order_test(solver, expected_order=4.0,
                                         step_sizes=step_sizes, tolerance=0.5)
        assert order > 3.5

    def test_etd4_allen_cahn_order(self):
        """Test ETD4 convergence order on Allen-Cahn equation (non-diagonal system)."""
        _, _, _, linear_op, nl_func = allen_cahn_setup()
        etd_config = ETDConfig(contour_points=64, contour_radius=20)
        solver = ETD4(lin_op=linear_op, nl_func=nl_func, etd_config=etd_config)
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        order, _ = allen_cahn_order_test(solver, expected_order=4.0,
                                                step_sizes=step_sizes, tolerance=1.0)
        # More lenient for stiff non-diagonal system
        assert order > 3.5

class TestETD5Order:
    """Test convergence order of the ETD5 solver."""
    
    def test_etd5_burgers_order(self):
        """Test ETD5 convergence order on Burgers equation (diagonal system)."""
        _, linear_op, nl_func = burgers_setup()
        solver = ETD5(lin_op=linear_op, nl_func=nl_func)
        step_sizes = [0.005, 0.004, 0.003, 0.002, 0.001]
        order, _ = burgers_order_test(solver, expected_order=5.0,
                                             step_sizes=step_sizes, tolerance=0.5)
        assert order > 4.5

    def test_etd5_kdv_order(self):
        """Test ETD4 convergence order on KdV equation."""
        _, linear_op, nl_func, _, _, _ = kdv_soliton_setup()
        etd_config = ETDConfig(modecutoff=0.0001, contour_points=64, contour_radius=2.0)
        solver = ETD5(lin_op=linear_op, nl_func=nl_func, etd_config=etd_config)
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        order, _ = kdv_order_test(solver, expected_order=5.0,
                                         step_sizes=step_sizes, tolerance=0.5)
        assert order > 4.5

    def test_etd5_allen_cahn_order(self):
        """Test ETD5 convergence order on Allen-Cahn equation (non-diagonal system)."""
        _, _, _, linear_op, nl_func = allen_cahn_setup()
        etd_config = ETDConfig(contour_points=64, contour_radius=10)
        solver = ETD5(lin_op=linear_op, nl_func=nl_func, etd_config=etd_config)
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        order, _ = allen_cahn_order_test(solver, expected_order=5.0,
                                                step_sizes=step_sizes, tolerance=1.5)
        assert order > 4.5

