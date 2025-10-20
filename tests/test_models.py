"""Model tests"""

import numpy as np
from rkstiff import models


def test_kdv_soliton_shape():
    x = np.linspace(-10, 10, 32)
    u = models.kdv_soliton(x, ampl=1.0, x0=0.0, t=0.0)
    assert u.shape == x.shape


def test_kdv_ops_and_burgers_ops():
    kx = np.linspace(0, 10, 8)
    L, NL = models.kdv_ops(kx)
    u = np.ones_like(kx)
    assert callable(NL)
    assert isinstance(L, np.ndarray)
    NL(u)
    L, NL = models.burgers_ops(kx, mu=0.1)
    NL(u)


def test_allen_cahn_ops():
    x = np.linspace(-1, 1, 8)
    Dx = np.eye(8)
    _, NL = models.allen_cahn_ops(x, Dx, epsilon=0.01)
    u = np.ones(6)
    NL(u)
