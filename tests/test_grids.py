"""Extra tests for rkstiff.grids to improve coverage."""

import pytest
from rkstiff import grids


def test_construct_x_kx_rfft_invalid_n():
    with pytest.raises(ValueError):
        grids.construct_x_kx_rfft(0)
    with pytest.raises(ValueError):
        grids.construct_x_kx_rfft(-5)


def test_construct_x_kx_rfft_valid():
    x, kx = grids.construct_x_kx_rfft(8)
    assert x.shape[0] == 8
    assert kx.shape[0] == 5


def test_construct_x_dx_cheb_invalid_n():
    with pytest.raises(ValueError):
        grids.construct_x_dx_cheb(1)


def test_construct_x_dx_cheb_valid():
    x, Dx = grids.construct_x_dx_cheb(8)
    assert x.shape[0] == 9
    assert Dx.shape == (9, 9)


def test_hankel_transform_invalid():
    from rkstiff.grids import HankelTransform

    with pytest.raises(ValueError):
        HankelTransform(nr=0, rmax=1.0)
    with pytest.raises(ValueError):
        HankelTransform(nr=10, rmax=0)


def test_hankel_transform_properties():
    from rkstiff.grids import HankelTransform

    ht = HankelTransform(nr=8, rmax=2.0)
    assert hasattr(ht, "r")
    assert hasattr(ht, "kr")
    assert callable(ht.ht)
    assert callable(ht.iht)
