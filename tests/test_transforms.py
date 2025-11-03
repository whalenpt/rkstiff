import numpy as np
import pytest
from rkstiff.transforms import HankelTransform


def test_hankel():
    """Test Hankel transform accuracy against known analytic transforms."""
    ht = HankelTransform(nr=50, rmax=2.0)
    a = 4

    f1 = np.exp(-(a**2) * ht.r**2)
    fsp1_ex = np.exp(-ht.kr**2 / (4 * a**2)) / (2 * a**2)
    fsp1 = ht.ht(f1)
    error1 = np.linalg.norm(fsp1 - fsp1_ex) / np.linalg.norm(fsp1_ex)
    assert error1 < 1e-10

    ht.rmax = 4
    sigma = 2
    w = 0.5
    f2 = np.exp(-sigma * ht.r**2) * np.sin(w * ht.r**2)
    omega = 1.0 / (4 * (sigma**2 + w**2))
    fsp2_ex = (
        -2
        * omega
        * np.exp(-sigma * omega * ht.kr**2)
        * (-w * np.cos(w * omega * ht.kr**2) + sigma * np.sin(w * omega * ht.kr**2))
    )
    fsp2 = ht.ht(f2)
    error2 = np.linalg.norm(fsp2 - fsp2_ex) / np.linalg.norm(fsp2_ex)
    assert error2 < 1e-10

    ht.nr = 25
    ht.rmax = 3
    f3 = np.exp(-a * ht.r)
    fsp3_ex = a * np.power(a**2 + ht.kr**2, -3.0 / 2)
    fsp3 = ht.ht(f3)
    error3 = np.linalg.norm(fsp3 - fsp3_ex) / np.linalg.norm(fsp3_ex)
    assert error3 < 1e-2


def test_hankel_transform_invalid():
    with pytest.raises(ValueError):
        HankelTransform(nr=0, rmax=1.0)
    with pytest.raises(ValueError):
        HankelTransform(nr=10, rmax=0)


def test_hankel_transform_properties():
    ht = HankelTransform(nr=8, rmax=2.0)
    assert hasattr(ht, "r")
    assert hasattr(ht, "kr")
    assert callable(ht.ht)
    assert callable(ht.iht)
