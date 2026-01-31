"""Comprehensive tests for correlation coefficient analysis.

Tests cover:
- Pearson correlation (linear)
- Spearman correlation (monotonic, rank-based)
- Kendall correlation (rank-based, tau-b)
- Edge cases (perfect correlation, no correlation, negative correlation)
- Both WaveformTrace and numpy array inputs
- Error handling for invalid methods
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.correlation import correlation_coefficient
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.statistical]


# Fixtures


@pytest.fixture
def linear_correlated_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create perfectly linearly correlated arrays.

    Returns:
        Tuple of (array1, array2) with perfect positive linear correlation
    """
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3  # Perfect linear relationship
    return x, y


@pytest.fixture
def negative_correlated_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create perfectly negatively correlated arrays.

    Returns:
        Tuple of (array1, array2) with perfect negative linear correlation
    """
    x = np.linspace(0, 10, 100)
    y = -2 * x + 3  # Perfect negative linear relationship
    return x, y


@pytest.fixture
def uncorrelated_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create uncorrelated arrays.

    Returns:
        Tuple of (array1, array2) with no correlation
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    return x, y


@pytest.fixture
def monotonic_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create monotonically related but nonlinear arrays.

    Returns:
        Tuple of (array1, array2) with perfect monotonic relationship
    """
    x = np.linspace(0, 10, 100)
    y = x**2  # Monotonic but nonlinear
    return x, y


@pytest.fixture
def simple_traces() -> tuple[WaveformTrace, WaveformTrace]:
    """Create simple WaveformTrace pair for testing.

    Returns:
        Tuple of (trace1, trace2) with positive correlation
    """
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    metadata1 = TraceMetadata(sample_rate=1e6, channel_name="trace1")
    metadata2 = TraceMetadata(sample_rate=1e6, channel_name="trace2")

    trace1 = WaveformTrace(metadata=metadata1, data=data1)
    trace2 = WaveformTrace(metadata=metadata2, data=data2)

    return trace1, trace2


# Test Pearson correlation


def test_pearson_perfect_positive_correlation(
    linear_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Pearson correlation with perfect positive correlation."""
    x, y = linear_correlated_arrays
    r = correlation_coefficient(x, y, method="pearson")

    assert r == pytest.approx(1.0, abs=1e-10)


def test_pearson_perfect_negative_correlation(
    negative_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Pearson correlation with perfect negative correlation."""
    x, y = negative_correlated_arrays
    r = correlation_coefficient(x, y, method="pearson")

    assert r == pytest.approx(-1.0, abs=1e-10)


def test_pearson_no_correlation(uncorrelated_arrays: tuple[np.ndarray, np.ndarray]) -> None:
    """Test Pearson correlation with uncorrelated data."""
    x, y = uncorrelated_arrays
    r = correlation_coefficient(x, y, method="pearson")

    # Should be close to 0 for truly uncorrelated data
    assert abs(r) < 0.2  # Allow some variation due to randomness


def test_pearson_default_method() -> None:
    """Test that Pearson is the default method."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    r_default = correlation_coefficient(x, y)
    r_pearson = correlation_coefficient(x, y, method="pearson")

    assert r_default == r_pearson


def test_pearson_with_traces(simple_traces: tuple[WaveformTrace, WaveformTrace]) -> None:
    """Test Pearson correlation with WaveformTrace inputs."""
    trace1, trace2 = simple_traces
    r = correlation_coefficient(trace1, trace2, method="pearson")

    # trace2 = 2 * trace1, so perfect positive correlation
    assert r == pytest.approx(1.0, abs=1e-10)


# Test Spearman correlation


def test_spearman_perfect_linear_correlation(
    linear_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Spearman correlation with perfect linear correlation."""
    x, y = linear_correlated_arrays
    rho = correlation_coefficient(x, y, method="spearman")

    assert rho == pytest.approx(1.0, abs=1e-10)


def test_spearman_perfect_monotonic_correlation(
    monotonic_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Spearman correlation with perfect monotonic (nonlinear) relationship."""
    x, y = monotonic_arrays
    rho = correlation_coefficient(x, y, method="spearman")

    # Spearman should detect perfect monotonic relationship even if nonlinear
    assert rho == pytest.approx(1.0, abs=1e-10)


def test_spearman_vs_pearson_nonlinear() -> None:
    """Test that Spearman captures monotonic relationship better than Pearson."""
    x = np.linspace(0, 10, 100)
    y = x**2  # Monotonic but nonlinear

    pearson_r = correlation_coefficient(x, y, method="pearson")
    spearman_rho = correlation_coefficient(x, y, method="spearman")

    # Spearman should be closer to 1 than Pearson for monotonic nonlinear data
    assert spearman_rho > pearson_r
    assert spearman_rho == pytest.approx(1.0, abs=1e-10)


def test_spearman_robust_to_outliers() -> None:
    """Test that Spearman is more robust to outliers than Pearson."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    # Original perfect correlation
    pearson_clean = correlation_coefficient(x, y, method="pearson")
    spearman_clean = correlation_coefficient(x, y, method="spearman")

    # Add outlier to y
    y_outlier = y.copy()
    y_outlier[-1] = 100.0  # Large outlier

    pearson_outlier = correlation_coefficient(x, y_outlier, method="pearson")
    spearman_outlier = correlation_coefficient(x, y_outlier, method="spearman")

    # Spearman should be less affected by outlier
    pearson_change = abs(pearson_clean - pearson_outlier)
    spearman_change = abs(spearman_clean - spearman_outlier)

    assert spearman_change < pearson_change


def test_spearman_negative_correlation(
    negative_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Spearman correlation with perfect negative correlation."""
    x, y = negative_correlated_arrays
    rho = correlation_coefficient(x, y, method="spearman")

    assert rho == pytest.approx(-1.0, abs=1e-10)


# Test Kendall correlation


def test_kendall_perfect_linear_correlation(
    linear_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Kendall correlation with perfect linear correlation."""
    x, y = linear_correlated_arrays
    tau = correlation_coefficient(x, y, method="kendall")

    assert tau == pytest.approx(1.0, abs=1e-10)


def test_kendall_perfect_monotonic_correlation(
    monotonic_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Kendall correlation with perfect monotonic (nonlinear) relationship."""
    x, y = monotonic_arrays
    tau = correlation_coefficient(x, y, method="kendall")

    # Kendall should detect perfect monotonic relationship even if nonlinear
    assert tau == pytest.approx(1.0, abs=1e-10)


def test_kendall_negative_correlation(
    negative_correlated_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Kendall correlation with perfect negative correlation."""
    x, y = negative_correlated_arrays
    tau = correlation_coefficient(x, y, method="kendall")

    assert tau == pytest.approx(-1.0, abs=1e-10)


def test_kendall_magnitude_vs_spearman() -> None:
    """Test that Kendall tau is typically smaller in magnitude than Spearman rho."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.5, 2.2, 3.1, 3.9, 5.2, 6.0, 6.8, 8.1, 9.3, 10.5])

    spearman_rho = correlation_coefficient(x, y, method="spearman")
    kendall_tau = correlation_coefficient(x, y, method="kendall")

    # Both should be positive and high
    assert spearman_rho > 0.9
    assert kendall_tau > 0.8

    # Kendall typically has smaller magnitude than Spearman
    # (but both detect same direction)
    assert np.sign(spearman_rho) == np.sign(kendall_tau)


# Test edge cases


def test_correlation_identical_arrays() -> None:
    """Test correlation of array with itself."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    pearson = correlation_coefficient(x, x, method="pearson")
    spearman = correlation_coefficient(x, x, method="spearman")
    kendall = correlation_coefficient(x, x, method="kendall")

    # All should be perfect correlation
    assert pearson == pytest.approx(1.0)
    assert spearman == pytest.approx(1.0)
    assert kendall == pytest.approx(1.0)


def test_correlation_constant_array() -> None:
    """Test correlation when one array is constant."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.full(5, 3.0)  # Constant

    # Correlation with constant array generates expected warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pearson = correlation_coefficient(x, y, method="pearson")

    # Correlation with constant should be nan (undefined)
    assert np.isnan(pearson)


def test_correlation_different_lengths() -> None:
    """Test correlation with different length arrays."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0])  # Shorter

    # Should truncate to shorter length
    r = correlation_coefficient(x, y, method="pearson")

    # Should compute correlation on first 3 elements
    assert np.isfinite(r)


def test_correlation_with_nans() -> None:
    """Test correlation with NaN values."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    pearson = correlation_coefficient(x, y, method="pearson")

    # Should return nan when input contains nan
    assert np.isnan(pearson)


def test_correlation_small_arrays() -> None:
    """Test correlation with very small arrays."""
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])

    pearson = correlation_coefficient(x, y, method="pearson")
    spearman = correlation_coefficient(x, y, method="spearman")
    kendall = correlation_coefficient(x, y, method="kendall")

    # All should work and give perfect correlation
    assert pearson == pytest.approx(1.0)
    assert spearman == pytest.approx(1.0)
    assert kendall == pytest.approx(1.0)


# Test error handling


def test_correlation_invalid_method() -> None:
    """Test correlation with invalid method raises ValueError."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    with pytest.raises(ValueError, match="Unknown correlation method"):
        correlation_coefficient(x, y, method="invalid")  # type: ignore[arg-type]


def test_correlation_method_case_sensitive() -> None:
    """Test that correlation method is case-sensitive."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    # Uppercase should fail
    with pytest.raises(ValueError, match="Unknown correlation method"):
        correlation_coefficient(x, y, method="PEARSON")  # type: ignore[arg-type]


# Test numerical properties


def test_correlation_range() -> None:
    """Test that all correlation methods return values in [-1, 1]."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)

    pearson = correlation_coefficient(x, y, method="pearson")
    spearman = correlation_coefficient(x, y, method="spearman")
    kendall = correlation_coefficient(x, y, method="kendall")

    assert -1 <= pearson <= 1
    assert -1 <= spearman <= 1
    assert -1 <= kendall <= 1


def test_correlation_symmetry() -> None:
    """Test that correlation is symmetric: corr(x,y) == corr(y,x)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 5.0, 7.0, 11.0])

    pearson_xy = correlation_coefficient(x, y, method="pearson")
    pearson_yx = correlation_coefficient(y, x, method="pearson")

    spearman_xy = correlation_coefficient(x, y, method="spearman")
    spearman_yx = correlation_coefficient(y, x, method="spearman")

    kendall_xy = correlation_coefficient(x, y, method="kendall")
    kendall_yx = correlation_coefficient(y, x, method="kendall")

    assert pearson_xy == pytest.approx(pearson_yx)
    assert spearman_xy == pytest.approx(spearman_yx)
    assert kendall_xy == pytest.approx(kendall_yx)


# Test real-world scenarios


def test_correlation_noisy_linear_relationship() -> None:
    """Test correlation with noisy linear relationship."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 100)
    noise = rng.normal(0, 0.5, 100)
    y = 2 * x + 3 + noise  # Linear with noise

    pearson = correlation_coefficient(x, y, method="pearson")
    spearman = correlation_coefficient(x, y, method="spearman")

    # Should still have high correlation
    assert pearson > 0.95
    assert spearman > 0.95


def test_correlation_exponential_relationship() -> None:
    """Test correlation with exponential relationship."""
    x = np.linspace(0, 5, 50)
    y = np.exp(x)

    pearson = correlation_coefficient(x, y, method="pearson")
    spearman = correlation_coefficient(x, y, method="spearman")

    # Spearman should be perfect (monotonic)
    # Pearson should be lower (nonlinear)
    assert spearman == pytest.approx(1.0, abs=1e-10)
    assert pearson < spearman


def test_correlation_with_ties() -> None:
    """Test correlation with tied values (important for Spearman/Kendall)."""
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    y = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0])

    # All methods should handle ties correctly
    pearson = correlation_coefficient(x, y, method="pearson")
    spearman = correlation_coefficient(x, y, method="spearman")
    kendall = correlation_coefficient(x, y, method="kendall")

    # All should be positive and high
    assert pearson > 0.8
    assert spearman > 0.8
    assert kendall > 0.7
