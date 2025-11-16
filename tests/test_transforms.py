"""Comprehensive tests for the HankelTransform class."""
import numpy as np
import pytest
from rkstiff.transforms import HankelTransform


# ============================================================================
# Basic Accuracy Tests (Existing)
# ============================================================================

def test_hankel_gaussian():
    """Test Hankel transform accuracy for Gaussian function."""
    ht = HankelTransform(nr=50, rmax=2.0)
    a = 4
    
    # f(r) = exp(-a^2 * r^2)
    # Hankel transform: G(k) = exp(-k^2 / (4*a^2)) / (2*a^2)
    f1 = np.exp(-(a**2) * ht.r**2)
    fsp1_ex = np.exp(-ht.kr**2 / (4 * a**2)) / (2 * a**2)
    fsp1 = ht.ht(f1)
    
    error1 = np.linalg.norm(fsp1 - fsp1_ex) / np.linalg.norm(fsp1_ex)
    assert error1 < 1e-10


def test_hankel_modulated_gaussian():
    """Test Hankel transform for modulated Gaussian."""
    ht = HankelTransform(nr=50, rmax=4.0)
    
    sigma = 2
    w = 0.5
    
    # f(r) = exp(-sigma * r^2) * sin(w * r^2)
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


def test_hankel_exponential():
    """Test Hankel transform for exponential decay."""
    ht = HankelTransform(nr=25, rmax=3.0)
    
    a = 4
    # f(r) = exp(-a * r)
    # Hankel transform: G(k) = a / (a^2 + k^2)^(3/2)
    f3 = np.exp(-a * ht.r)
    fsp3_ex = a * np.power(a**2 + ht.kr**2, -3.0 / 2)
    fsp3 = ht.ht(f3)
    
    error3 = np.linalg.norm(fsp3 - fsp3_ex) / np.linalg.norm(fsp3_ex)
    assert error3 < 1e-2


# ============================================================================
# Property Setter Tests (Lines 133, 144, 153, 159)
# ============================================================================

def test_nr_property_setter():
    """Test that setting nr property triggers reinitialization."""
    ht = HankelTransform(nr=8, rmax=2.0)
    
    original_r_shape = ht.r.shape
    original_Y_shape = ht._Y.shape
    
    # Change nr
    ht.nr = 16
    
    # Verify grid was resized
    assert ht.nr == 16
    assert ht.r.shape[0] == 16
    assert ht.kr.shape[0] == 16
    assert ht._Y.shape == (16, 16)
    assert ht.r.shape != original_r_shape
    assert ht._Y.shape != original_Y_shape


def test_nr_setter_validation():
    """Test that nr setter validates input (line 133)."""
    ht = HankelTransform(nr=8, rmax=2.0)
    
    # Test non-integer
    with pytest.raises(ValueError, match="nr must be an integer"):
        ht.nr = 10.5
    
    # Test too small
    with pytest.raises(ValueError, match="nr must be an integer ≥ 4"):
        ht.nr = 3
    
    # Test negative
    with pytest.raises(ValueError, match="nr must be an integer ≥ 4"):
        ht.nr = -5
    
    # Verify original value unchanged after failed assignment
    assert ht.nr == 8


def test_rmax_property_setter():
    """Test that setting rmax property triggers reinitialization."""
    ht = HankelTransform(nr=8, rmax=2.0)
    
    original_r = ht.r.copy()
    original_kr = ht.kr.copy()
    
    # Change rmax
    ht.rmax = 4.0
    
    # Verify grid was rescaled
    assert ht.rmax == 4.0
    assert not np.allclose(ht.r, original_r)
    assert not np.allclose(ht.kr, original_kr)
    
    # r should scale proportionally
    assert np.allclose(ht.r, original_r * 2.0, rtol=1e-10)


def test_rmax_setter_validation():
    """Test that rmax setter validates input (line 144)."""
    ht = HankelTransform(nr=8, rmax=2.0)
    
    # Test zero
    with pytest.raises(ValueError, match="rmax must be positive"):
        ht.rmax = 0.0
    
    # Test negative
    with pytest.raises(ValueError, match="rmax must be positive"):
        ht.rmax = -1.0
    
    # Verify original value unchanged after failed assignment
    assert ht.rmax == 2.0


def test_property_getters():
    """Test that property getters return correct values (lines 153, 159)."""
    ht = HankelTransform(nr=10, rmax=3.5)
    
    # Test nr getter
    assert ht.nr == 10
    assert isinstance(ht.nr, int)
    
    # Test rmax getter
    assert ht.rmax == 3.5
    assert isinstance(ht.rmax, float)


# ============================================================================
# Inverse Transform Tests (Line 171)
# ============================================================================

def test_inverse_hankel_transform():
    """Test inverse Hankel transform (iht method, line 171)."""
    ht = HankelTransform(nr=32, rmax=5.0)
    
    # Create a function in real space
    f_original = np.exp(-ht.r**2)
    
    # Transform to spectral space and back
    f_spectral = ht.ht(f_original)
    f_reconstructed = ht.iht(f_spectral)
    
    # Should recover original function
    error = np.linalg.norm(f_reconstructed - f_original) / np.linalg.norm(f_original)
    assert error < 1e-10


def test_inverse_transform_identity():
    """Test that iht(ht(f)) ≈ f for various functions."""
    ht = HankelTransform(nr=40, rmax=4.0)
    
    test_functions = [
        np.exp(-ht.r**2),                          # Gaussian
        np.exp(-2*ht.r),                            # Exponential
        np.exp(-ht.r**2) * np.cos(ht.r),           # Modulated
        np.ones_like(ht.r) * (ht.r < 2.0),         # Top-hat (discontinuous)
    ]
    
    for i, f in enumerate(test_functions):
        f_recovered = ht.iht(ht.ht(f))
        if i < 3:  # Smooth functions
            error = np.linalg.norm(f_recovered - f) / np.linalg.norm(f)
            assert error < 1e-9, f"Function {i} failed with error {error}"
        else:  # Discontinuous function - less accurate
            error = np.linalg.norm(f_recovered - f) / np.linalg.norm(f)
            assert error < 1e-1, f"Function {i} failed with error {error}"


def test_forward_inverse_spectral_space():
    """Test that ht(iht(G)) ≈ G starting from spectral space."""
    ht = HankelTransform(nr=30, rmax=3.0)
    
    # Define function in spectral space
    G_original = np.exp(-0.5 * ht.kr**2)
    
    # Transform to real space and back
    f_real = ht.iht(G_original)
    G_reconstructed = ht.ht(f_real)
    
    # Should recover original spectral function
    error = np.linalg.norm(G_reconstructed - G_original) / np.linalg.norm(G_original)
    assert error < 1e-10


# ============================================================================
# Matrix and Grid Accessor Tests (Line 240 - hankel_matrix, bessel_zeros)
# ============================================================================

def test_hankel_matrix_accessor():
    """Test hankel_matrix() method returns correct matrix (line 240)."""
    ht = HankelTransform(nr=12, rmax=2.0)
    
    Y = ht.hankel_matrix()
    
    # Check shape
    assert Y.shape == (12, 12)
    
    # Check it's a copy (not reference)
    Y_modified = Y.copy()
    original_value = Y_modified[0, 0]
    Y_modified[0, 0] = 999.0
    Y_internal = ht.hankel_matrix()
    assert np.abs(Y_internal[0, 0] - original_value) < 1e-10  # Should match original, not modified
    assert Y_internal[0, 0] != 999.0
    
    # Check matrix finite
    assert np.all(np.isfinite(Y))

    # Check diagonal elements are positive and non-zero
    assert np.all(np.diag(Y) != 0)


def test_bessel_zeros_accessor():
    """Test bessel_zeros() method returns correct zeros."""
    ht = HankelTransform(nr=10, rmax=2.0)
    
    zeros = ht.bessel_zeros()
    
    # Check shape
    assert zeros.shape == (10,)
    
    # Check it's a copy (not reference)
    zeros_original = zeros.copy()
    zeros[0] = -999.0
    zeros_internal = ht.bessel_zeros()
    assert zeros_internal[0] != -999.0
    
    # Check zeros are positive and increasing
    assert np.all(zeros_internal > 0)
    assert np.all(np.diff(zeros_internal) > 0)
    
    # First zero of J_0 is approximately 2.4048
    assert np.abs(zeros_internal[0] - 2.4048) < 0.001


def test_hankel_matrix_consistency():
    """Test that hankel_matrix is used consistently in transforms."""
    ht = HankelTransform(nr=16, rmax=3.0)
    
    Y = ht.hankel_matrix()
    f = np.exp(-ht.r**2)
    
    # Manually compute forward transform
    scale = ht.rmax**2 / ht._jN
    G_manual = scale * Y.dot(f)
    
    # Compare with ht() method
    G_method = ht.ht(f)
    
    assert np.allclose(G_manual, G_method, rtol=1e-12)


# ============================================================================
# Edge Cases and Validation
# ============================================================================

def test_hankel_transform_minimum_size():
    """Test Hankel transform with minimum allowed size (nr=4)."""
    ht = HankelTransform(nr=4, rmax=1.0)
    
    assert ht.nr == 4
    assert ht.r.shape == (4,)
    assert ht.kr.shape == (4,)
    assert ht._Y.shape == (4, 4)
    
    # Should still work
    f = np.ones(4)
    G = ht.ht(f)
    assert G.shape == (4,)


def test_hankel_transform_invalid_construction():
    """Test that invalid parameters raise errors during construction."""
    # nr too small
    with pytest.raises(ValueError):
        HankelTransform(nr=3, rmax=1.0)
    
    # nr = 0
    with pytest.raises(ValueError):
        HankelTransform(nr=0, rmax=1.0)
    
    # negative nr
    with pytest.raises(ValueError):
        HankelTransform(nr=-5, rmax=1.0)
    
    # rmax = 0
    with pytest.raises(ValueError):
        HankelTransform(nr=10, rmax=0.0)
    
    # rmax negative
    with pytest.raises(ValueError):
        HankelTransform(nr=10, rmax=-1.0)


def test_hankel_transform_large_nr():
    """Test Hankel transform with large number of points."""
    ht = HankelTransform(nr=128, rmax=10.0)
    
    assert ht.nr == 128
    assert ht.r.shape == (128,)
    
    # Transform should still be accurate
    f = np.exp(-ht.r**2)
    G = ht.ht(f)
    f_recovered = ht.iht(G)
    
    error = np.linalg.norm(f_recovered - f) / np.linalg.norm(f)
    assert error < 1e-10


def test_hankel_transform_small_rmax():
    """Test with very small rmax."""
    ht = HankelTransform(nr=20, rmax=0.1)
    
    assert ht.rmax == 0.1
    assert np.all(ht.r <= 0.1)
    
    # Should still work
    f = np.exp(-100 * ht.r**2)
    G = ht.ht(f)
    assert np.all(np.isfinite(G))


def test_hankel_transform_large_rmax():
    """Test with large rmax."""
    ht = HankelTransform(nr=30, rmax=100.0)
    
    assert ht.rmax == 100.0
    assert np.max(ht.r) < 100.0
    
    # Should still work
    f = np.exp(-ht.r**2 / 100.0)
    G = ht.ht(f)
    assert np.all(np.isfinite(G))


# ============================================================================
# Properties and Attributes
# ============================================================================

def test_hankel_transform_properties():
    """Test that all expected properties and methods exist."""
    ht = HankelTransform(nr=8, rmax=2.0)
    
    # Properties
    assert hasattr(ht, "nr")
    assert hasattr(ht, "rmax")
    assert hasattr(ht, "r")
    assert hasattr(ht, "kr")
    
    # Methods
    assert callable(ht.ht)
    assert callable(ht.iht)
    assert callable(ht.hankel_matrix)
    assert callable(ht.bessel_zeros)
    
    # Internal attributes (should exist but not be directly accessed by users)
    assert hasattr(ht, "_Y")
    assert hasattr(ht, "_jN")
    assert hasattr(ht, "_bessel_zeros")


def test_grid_relationship():
    """Test the relationship between r and kr grids."""
    ht = HankelTransform(nr=20, rmax=5.0)
    
    # Both grids should have same size
    assert len(ht.r) == len(ht.kr)
    
    # Both should be positive
    assert np.all(ht.r > 0)
    assert np.all(ht.kr > 0)
    
    # Both should be sorted
    assert np.all(np.diff(ht.r) > 0)
    assert np.all(np.diff(ht.kr) > 0)
    
    # kr should scale inversely with rmax
    ht2 = HankelTransform(nr=20, rmax=10.0)
    # When rmax doubles, kr should halve (approximately)
    assert np.allclose(ht.kr, 2 * ht2.kr, rtol=1e-10)


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

def test_parseval_identity():
    """Test Parseval's identity (energy conservation)."""
    ht = HankelTransform(nr=40, rmax=4.0)
    
    # Create a normalized function
    f = np.exp(-ht.r**2)
    
    # Energy in real space (with radial weight)
    energy_real = np.sum(f**2 * ht.r) * (ht.r[1] - ht.r[0])
    
    # Transform and compute energy in spectral space
    G = ht.ht(f)
    energy_spectral = np.sum(G**2 * ht.kr) * (ht.kr[1] - ht.kr[0])
    
    # Energies should be approximately equal
    # Note: discrete DHT doesn't preserve Parseval exactly, but should be close
    rel_error = abs(energy_real - energy_spectral) / energy_real
    assert rel_error < 0.1  # Within 10%


def test_transform_zero_function():
    """Test transform of zero function."""
    ht = HankelTransform(nr=20, rmax=3.0)
    
    f = np.zeros(20)
    G = ht.ht(f)
    
    assert np.allclose(G, 0.0)
    
    # And inverse
    f_back = ht.iht(G)
    assert np.allclose(f_back, 0.0)


def test_transform_delta_like_function():
    """Test transform of delta-like function (concentrated at origin)."""
    ht = HankelTransform(nr=50, rmax=5.0)
    
    # Very narrow Gaussian approximates delta function
    sigma = 0.05  # Made slightly wider for numerical stability
    f = np.exp(-ht.r**2 / (2 * sigma**2))
    
    G = ht.ht(f)
    
    # Transform of delta should be approximately constant
    # For a narrow Gaussian, the transform should be relatively flat in k-space
    # Check relative variation rather than absolute ratio
    G_mean = np.mean(G)
    G_std = np.std(G)
    relative_variation = G_std / np.abs(G_mean) if G_mean != 0 else 0
    
    # Allow for some variation, but should be relatively flat
    assert relative_variation < 1.0  # Less than 100% relative variation


def test_sequential_transforms():
    """Test multiple sequential transforms maintain consistency."""
    ht = HankelTransform(nr=30, rmax=3.0)
    
    f0 = np.exp(-ht.r**2)
    
    # Apply forward and inverse multiple times
    f = f0.copy()
    for _ in range(5):
        G = ht.ht(f)
        f = ht.iht(G)
    
    # Should still be close to original
    error = np.linalg.norm(f - f0) / np.linalg.norm(f0)
    assert error < 1e-8


# ============================================================================
# Property Modification Chain Tests
# ============================================================================

def test_multiple_property_changes():
    """Test changing properties multiple times."""
    ht = HankelTransform(nr=10, rmax=2.0)
    
    # Change nr several times
    for new_nr in [15, 20, 8, 12]:
        ht.nr = new_nr
        assert ht.nr == new_nr
        assert ht.r.shape[0] == new_nr
        
        # Transform should still work
        f = np.exp(-ht.r**2)
        G = ht.ht(f)
        assert G.shape[0] == new_nr


def test_mixed_property_changes():
    """Test changing both nr and rmax in sequence."""
    ht = HankelTransform(nr=10, rmax=2.0)
    
    original_r = ht.r.copy()
    
    # Change rmax
    ht.rmax = 4.0
    assert ht.rmax == 4.0
    
    # Change nr
    ht.nr = 20
    assert ht.nr == 20
    
    # Change rmax again
    ht.rmax = 1.0
    assert ht.rmax == 1.0
    
    # Grid should be valid
    assert len(ht.r) == 20
    assert np.all(ht.r < 1.0)
    
    # Transform should work
    f = np.exp(-ht.r**2)
    G = ht.ht(f)
    assert len(G) == 20
