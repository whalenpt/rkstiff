import numpy as np
import pytest

from rkstiff.etd import (
    ETDConfig,
    psi1,
    psi2,
    psi3,
    ETDAS,
    ETDCS,
)
from rkstiff.solveras import SolverConfig


# =============================================================================
# Dummy subclasses so ETDAS and ETDCS become instantiable
# =============================================================================

class DummyETDAS(ETDAS):
    """Concrete minimal subclass of ETDAS providing required abstract methods."""

    def __init__(self, lin_op, nl_func, *args, **kwargs):
        super().__init__(lin_op, nl_func, *args, **kwargs)

        # BaseSolverAS expects these fields to exist:
        self.lin = lin_op
        self.nl = nl_func
        self.u = np.zeros(lin_op.shape[0])

    # Required abstract methods
    def _q(self, u):
        return u

    def _reset(self):
        pass

    def _update_stages(self, *args, **kwargs):
        pass


class DummyETDCS(ETDCS):
    """Concrete minimal subclass of ETDCS providing required abstract methods."""

    def __init__(self, lin_op, nl_func, *args, **kwargs):
        super().__init__(lin_op, nl_func, *args, **kwargs)

        # BaseSolverCS expects:
        self.lin = lin_op
        self.nl = nl_func
        self.u = np.zeros(lin_op.shape[0])

    # Required abstract methods
    def _reset(self):
        pass

    def _update_stages(self, *args, **kwargs):
        pass


def dummy_nl(u):
    return u * 0.0


# =============================================================================
# ETDConfig Tests
# =============================================================================

def test_etdconfig_modecutoff_valid():
    cfg = ETDConfig(modecutoff=0.5)
    assert cfg.modecutoff == 0.5


def test_etdconfig_modecutoff_invalid():
    with pytest.raises(ValueError):
        ETDConfig(modecutoff=1.5)
    with pytest.raises(ValueError):
        ETDConfig(modecutoff=0.0)


def test_etdconfig_contour_points_valid():
    cfg = ETDConfig(contour_points=8)
    assert cfg.contour_points == 8


def test_etdconfig_contour_points_invalid_type():
    with pytest.raises(TypeError):
        ETDConfig(contour_points=3.14)


def test_etdconfig_contour_points_invalid_value():
    with pytest.raises(ValueError):
        ETDConfig(contour_points=1)


def test_etdconfig_contour_radius_valid():
    cfg = ETDConfig(contour_radius=2.5)
    assert cfg.contour_radius == 2.5


def test_etdconfig_contour_radius_invalid():
    with pytest.raises(ValueError):
        ETDConfig(contour_radius=0.0)
    with pytest.raises(ValueError):
        ETDConfig(contour_radius=-1.0)


# =============================================================================
# Psi Function Tests
# =============================================================================

def test_psi1_basic():
    z = np.array([1.0, 2.0])
    expected = (np.exp(z) - 1) / z
    assert np.allclose(psi1(z), expected)


def test_psi2_basic():
    z = np.array([1.0, 3.0])
    expected = 2 * (np.exp(z) - 1 - z) / (z**2)
    assert np.allclose(psi2(z), expected)


def test_psi3_basic():
    z = np.array([1.0, 3.0])
    expected = 6 * (np.exp(z) - 1 - z - z**2 / 2) / (z**3)
    assert np.allclose(psi3(z), expected)


def test_psi_small_z_stability():
    """Ensure psi-functions remain finite near z → 0."""
    z = np.array([1e-8, -1e-8])
    assert np.all(np.isfinite(psi1(z)))
    assert np.all(np.isfinite(psi2(z)))
    assert np.all(np.isfinite(psi3(z)))


def test_psi_broadcasting():
    z = np.ones((2, 2))
    assert psi1(z).shape == (2, 2)
    assert psi2(z).shape == (2, 2)
    assert psi3(z).shape == (2, 2)


# =============================================================================
# ETDAS Tests (using DummyETDAS)
# =============================================================================

def test_etdas_initialization_defaults():
    L = np.eye(3)
    solver = DummyETDAS(L, dummy_nl)

    assert isinstance(solver.etd_config, ETDConfig)
    assert solver._h_coeff is None
    assert solver.nl is dummy_nl
    assert solver.lin.shape == (3, 3)
    assert solver.u.shape == (3,)


def test_etdas_initialization_custom_config():
    L = np.eye(2)
    cfg = SolverConfig()
    etd_cfg = ETDConfig(modecutoff=0.5, contour_points=16, contour_radius=1.5)

    solver = DummyETDAS(L, dummy_nl, config=cfg, etd_config=etd_cfg)

    assert solver.config is cfg
    assert solver.etd_config is etd_cfg
    assert solver._h_coeff is None
    assert solver.u.shape == (2,)


# =============================================================================
# ETDCS Tests (using DummyETDCS)
# =============================================================================

def test_etdcs_initialization():
    L = np.eye(4)
    solver = DummyETDCS(L, dummy_nl)

    assert isinstance(solver.etd_config, ETDConfig)
    assert solver._h_coeff is None
    assert solver.lin.shape == (4, 4)
    assert solver.u.shape == (4,)
    assert solver.nl is dummy_nl
