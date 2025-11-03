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
