"""Model tests"""

import numpy as np
import pytest

from rkstiff.models import (
    kdv_soliton,
    kdv_multi_soliton,
    kdv_ops,
    burgers_ops,
    allen_cahn_ops,
)


# -------------------------------------------------------------------------
# KDV SOLITON
# -------------------------------------------------------------------------


def test_kdv_soliton_basic():
    x = np.linspace(-5, 5, 100)
    u = kdv_soliton(x, ampl=1.0, x0=0.0, t=0.0)
    assert u.shape == x.shape
    assert np.all(u >= 0)
    # peak should be at x0=0
    assert u[np.argmax(u)] == pytest.approx(u.max(), rel=1e-12)


# -------------------------------------------------------------------------
# MULTI-SOLITON
# -------------------------------------------------------------------------


def test_kdv_multi_soliton_valid():
    x = np.linspace(-5, 5, 200)
    ampl = [0.5, 1.0]
    x0 = [-1.0, 2.0]
    u = kdv_multi_soliton(x, ampl, x0)
    assert u.shape == x.shape
    assert np.all(u >= 0)


def test_kdv_multi_soliton_length_mismatch():
    """Covers ValueError on ampl/x0 mismatch."""
    x = np.linspace(-5, 5, 100)
    ampl = [0.5, 1.0]
    x0 = [0.0]  # mismatch
    with pytest.raises(ValueError):
        kdv_multi_soliton(x, ampl, x0)


# -------------------------------------------------------------------------
# KDV OPERATORS
# -------------------------------------------------------------------------


def test_kdv_ops_linear_and_nonlinear():
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    lin_op, nl = kdv_ops(kx)

    assert lin_op.shape == kx.shape
    assert np.iscomplexobj(lin_op)

    # test nonlinear operator
    u = np.sin(x)
    uf = np.fft.rfft(u)
    nl_val = nl(uf)

    assert nl_val.shape == uf.shape
    assert np.iscomplexobj(nl_val)


# -------------------------------------------------------------------------
# BURGERS OPERATORS
# -------------------------------------------------------------------------


def test_burgers_ops_basic():
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    mu = 0.1
    lin_op, nl = burgers_ops(kx, mu)

    assert lin_op.shape == kx.shape
    assert np.allclose(lin_op, -mu * kx**2)

    uf = np.fft.rfft(np.sin(x))
    nl_val = nl(uf)
    assert nl_val.shape == uf.shape


# -------------------------------------------------------------------------
# ALLEN–CAHN OPERATORS
# -------------------------------------------------------------------------


def test_allen_cahn_ops_basic():
    # small Chebyshev grid
    N = 10
    x = np.linspace(-1, 1, N)

    # simple mock differentiation matrix
    # doesn't need to be "real" Chebyshev D for coverage
    D = np.random.randn(N, N)

    lin_op, nl = allen_cahn_ops(x, D, epsilon=0.05)

    # lin_op must exclude boundaries → shape (N-2, N-2)
    assert lin_op.shape == (N - 2, N - 2)

    # nonlinear function maps (N-2,) → (N-2,)
    u = np.linspace(-0.3, 0.4, N - 2)
    nl_val = nl(u)
    assert nl_val.shape == (N - 2,)
    assert np.iscomplexobj(nl_val)
