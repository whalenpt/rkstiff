import numpy as np
import pytest
from rkstiff import derivatives


def test_dx_rfft_empty():
    arr = np.array([])
    kx = np.array([])
    out = derivatives.dx_rfft(arr, kx)
    assert out.size == 0


def test_dx_rfft_real():
    """Test dx_rfft with real-valued input."""
    N = 8
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    arr = np.sin(x)
    out = derivatives.dx_rfft(kx, arr, n=1)

    assert out.shape == arr.shape
    assert not np.iscomplexobj(out)
    expected = np.cos(x)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_dx_rfft_complex():
    """Test that dx_rfft raises TypeError for complex input."""
    arr = np.array([1.0, 2.0, 3.0])
    kx = np.array([0.0, 1.0])
    arr_c = arr + 1j * arr

    with pytest.raises(TypeError):
        derivatives.dx_rfft(kx, arr_c)


def test_dx_rfft_shape_mismatch():
    """Test that dx_rfft validates kx shape."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    kx_wrong = np.array([0.0, 1.0])  # Wrong size (should be 3 for N=4)

    with pytest.raises(ValueError):
        derivatives.dx_rfft(kx_wrong, arr)


def test_dx_rfft_second_derivative():
    """Test second derivative computation."""
    N = 16
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    arr = np.sin(x)
    out = derivatives.dx_rfft(kx, arr, n=2)

    expected = -np.sin(x)
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_dx_rfft_zero_derivative():
    """Test that n=0 returns the input unchanged."""
    N = 8
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    kx = np.fft.rfftfreq(N, d=1.0)

    out = derivatives.dx_rfft(kx, arr, n=0)
    np.testing.assert_array_equal(out, arr)


def test_dx_rfft_negative_n():
    """Test that negative n raises ValueError."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="non-negative"):
        derivatives.dx_rfft(kx, arr, n=-1)


def test_dx_rfft_non_integer_n():
    """Test that non-integer n raises TypeError."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError):
        derivatives.dx_rfft(kx, arr, n=1.5)


def test_dx_rfft_real():
    """Test dx_rfft with real-valued input."""
    N = 8
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    arr = np.sin(x)
    out = derivatives.dx_rfft(kx, arr, n=1)

    assert out.shape == arr.shape
    assert not np.iscomplexobj(out)
    expected = np.cos(x)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_dx_rfft_complex():
    """Test that dx_rfft raises TypeError for complex input."""
    arr = np.array([1.0, 2.0, 3.0])
    kx = np.array([0.0, 1.0])
    arr_c = arr + 1j * arr

    with pytest.raises(TypeError):
        derivatives.dx_rfft(kx, arr_c)


def test_dx_rfft_second_derivative():
    """Test second derivative computation."""
    N = 16
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    arr = np.sin(x)
    out = derivatives.dx_rfft(kx, arr, n=2)

    expected = -np.sin(x)
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_dx_rfft_zero_derivative():
    """Test that n=0 returns the input unchanged."""
    N = 8
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    kx = np.fft.rfftfreq(N, d=1.0)

    out = derivatives.dx_rfft(kx, arr, n=0)
    np.testing.assert_array_equal(out, arr)


def test_dx_rfft_negative_n():
    """Test that negative n raises ValueError."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        derivatives.dx_rfft(kx, arr, n=-1)


def test_dx_rfft_non_integer_n():
    """Test that non-integer n raises TypeError."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError):
        derivatives.dx_rfft(kx, arr, n=1.5)

def test_dx_rfft_non_integer_order():
    """dx_rfft should reject non-integer derivative orders."""
    # N = 4 → rfft output length is 3 → choose matching kx
    u = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="must be an integer"):
        derivatives.dx_rfft(kx, u, n=1.5)

def test_dx_rfft_negative_order():
    """dx_rfft should reject negative derivative orders."""
    u = np.array([1.0, 2.0, 3.0, 4.0])
    kx = np.array([0.0, 1.0, 2.0])  # rFFT shape for N=4

    with pytest.raises(ValueError, match="non-negative"):
        derivatives.dx_rfft(kx, u, n=-1)
