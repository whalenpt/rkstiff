"""Tests for rkstiff.grids."""

import numpy as np
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


def test_construct_x_kx_rfft_typeerror():
    with pytest.raises(TypeError):
        grids.construct_x_kx_rfft(8.5)


def test_construct_x_kx_rfft_odd_n():
    with pytest.raises(ValueError):
        grids.construct_x_kx_rfft(7)


def test_construct_x_kx_fft_typeerror():
    with pytest.raises(TypeError):
        grids.construct_x_kx_fft("10")


def test_construct_x_kx_fft_odd_n():
    with pytest.raises(ValueError):
        grids.construct_x_kx_fft(5)


def test_construct_x_kx_fft_valid():
    x, kx = grids.construct_x_kx_fft(8)
    assert x.shape == (8,)
    assert kx.shape == (8,)


def test_construct_x_cheb_typeerror():
    with pytest.raises(TypeError):
        grids.construct_x_cheb("10")


def test_construct_x_cheb_small_n():
    with pytest.raises(ValueError):
        grids.construct_x_cheb(1)

def test_mirror_grid_basic():
    r = np.array([0, 1, 2, 3])
    u = np.array([10, 20, 30, 40])
    rnew, unew = grids.mirror_grid(r, u)
    # rnew should be symmetric
    assert np.all(rnew[:4] == -np.flipud(r))
    assert np.all(rnew[4:] == r)
    # unew should be symmetric
    assert np.all(unew[:4] == np.flipud(u))
    assert np.all(unew[4:] == u)


def test_mirror_grid_only_r():
    r = np.array([0, 1, 2])
    rnew, unew = grids.mirror_grid(r)
    assert np.all(rnew[:3] == -np.flipud(r))
    assert np.all(rnew[3:] == r)
    assert unew is None


def test_mirror_grid_axis_0():
    r = np.array([0, 1, 2])
    u = np.array([[1, 2, 3], [4, 5, 6]])
    rnew, unew = grids.mirror_grid(r, u, axis=0)
    assert unew.shape[0] == 4


def test_mirror_grid_axis_1():
    r = np.array([0, 1, 2])
    u = np.array([[1, 2, 3], [4, 5, 6]])
    rnew, unew = grids.mirror_grid(r, u, axis=1)
    assert unew.shape[1] == 6


def test_mirror_grid_invalid_axis():
    r = np.array([0, 1, 2])
    u = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        grids.mirror_grid(r, u, axis=99)