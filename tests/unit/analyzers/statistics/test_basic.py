"""Comprehensive tests for basic statistical analysis functions.

Tests cover:
- Basic statistics (mean, var, std, min, max, range)
- Percentiles and quartiles
- Weighted mean
- Running statistics
- Summary statistics
- Edge cases (empty, single value, uniform data)
- Both WaveformTrace and numpy array inputs
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.basic import (
    basic_stats,
    measure,
    percentiles,
    quartiles,
    running_stats,
    summary_stats,
    weighted_mean,
    weighted_std,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit]


# Fixtures


@pytest.fixture
def simple_array() -> np.ndarray:
    """Create simple numpy array for testing.

    Returns:
        Array with known statistical properties
    """
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def simple_trace(simple_array: np.ndarray) -> WaveformTrace:
    """Create simple WaveformTrace for testing.

    Args:
        simple_array: Input array fixture

    Returns:
        WaveformTrace with known statistical properties
    """
    metadata = TraceMetadata(
        sample_rate=1e6,
        channel_name="test_trace",
    )
    return WaveformTrace(metadata=metadata, data=simple_array)


@pytest.fixture
def uniform_array() -> np.ndarray:
    """Create uniform array (all same values).

    Returns:
        Array with all values equal to 5.0
    """
    return np.full(100, 5.0)


@pytest.fixture
def gaussian_array() -> np.ndarray:
    """Create gaussian-distributed array.

    Returns:
        Array with mean=0, std=1
    """
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=1000)


# Test basic_stats


def test_basic_stats_with_array(simple_array: np.ndarray) -> None:
    """Test basic_stats with numpy array."""
    result = basic_stats(simple_array)

    assert result["mean"] == pytest.approx(3.0)
    assert result["variance"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(np.sqrt(2.0))
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["range"] == 4.0
    assert result["count"] == 5


def test_basic_stats_with_trace(simple_trace: WaveformTrace) -> None:
    """Test basic_stats with WaveformTrace."""
    result = basic_stats(simple_trace)

    assert result["mean"] == pytest.approx(3.0)
    assert result["count"] == 5


def test_basic_stats_ddof_parameter() -> None:
    """Test basic_stats with different ddof values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # ddof=0 (default, population variance)
    result0 = basic_stats(data, ddof=0)
    assert result0["variance"] == pytest.approx(2.0)

    # ddof=1 (sample variance)
    result1 = basic_stats(data, ddof=1)
    assert result1["variance"] == pytest.approx(2.5)


def test_basic_stats_uniform_data(uniform_array: np.ndarray) -> None:
    """Test basic_stats with uniform data."""
    result = basic_stats(uniform_array)

    assert result["mean"] == pytest.approx(5.0)
    assert result["variance"] == pytest.approx(0.0)
    assert result["std"] == pytest.approx(0.0)
    assert result["min"] == 5.0
    assert result["max"] == 5.0
    assert result["range"] == 0.0


def test_basic_stats_single_value() -> None:
    """Test basic_stats with single value."""
    data = np.array([42.0])
    result = basic_stats(data)

    assert result["mean"] == 42.0
    assert result["variance"] == 0.0
    assert result["min"] == 42.0
    assert result["max"] == 42.0
    assert result["count"] == 1


def test_basic_stats_negative_values() -> None:
    """Test basic_stats with negative values."""
    data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    result = basic_stats(data)

    assert result["mean"] == pytest.approx(0.0)
    assert result["min"] == -5.0
    assert result["max"] == 5.0
    assert result["range"] == 10.0


# Test percentiles


def test_percentiles_default_quartiles(simple_array: np.ndarray) -> None:
    """Test percentiles with default quartiles."""
    result = percentiles(simple_array)

    assert result["p0"] == 1.0
    assert result["p25"] == 2.0
    assert result["p50"] == 3.0
    assert result["p75"] == 4.0
    assert result["p100"] == 5.0


def test_percentiles_custom_values(simple_array: np.ndarray) -> None:
    """Test percentiles with custom values."""
    result = percentiles(simple_array, p=[10, 90])

    assert "p10" in result
    assert "p90" in result
    assert result["p10"] < result["p90"]


def test_percentiles_with_trace(simple_trace: WaveformTrace) -> None:
    """Test percentiles with WaveformTrace."""
    result = percentiles(simple_trace, p=[50])

    assert result["p50"] == 3.0


def test_percentiles_decimal_values() -> None:
    """Test percentiles with decimal percentile values."""
    data = np.linspace(0, 100, 1000)
    result = percentiles(data, p=[2.5, 97.5])

    assert "p2.5" in result
    assert "p97.5" in result
    assert result["p2.5"] < result["p97.5"]


def test_percentiles_single_value() -> None:
    """Test percentiles with single value."""
    data = np.array([42.0])
    result = percentiles(data, p=[0, 50, 100])

    assert result["p0"] == 42.0
    assert result["p50"] == 42.0
    assert result["p100"] == 42.0


# Test quartiles


def test_quartiles_standard_case(simple_array: np.ndarray) -> None:
    """Test quartiles with standard data."""
    result = quartiles(simple_array)

    assert result["q1"] == 2.0
    assert result["median"] == 3.0
    assert result["q3"] == 4.0
    assert result["iqr"] == 2.0
    assert result["lower_fence"] == -1.0
    assert result["upper_fence"] == 7.0


def test_quartiles_with_trace(simple_trace: WaveformTrace) -> None:
    """Test quartiles with WaveformTrace."""
    result = quartiles(simple_trace)

    assert result["median"] == 3.0
    assert result["iqr"] == 2.0


def test_quartiles_uniform_data(uniform_array: np.ndarray) -> None:
    """Test quartiles with uniform data."""
    result = quartiles(uniform_array)

    assert result["q1"] == 5.0
    assert result["median"] == 5.0
    assert result["q3"] == 5.0
    assert result["iqr"] == 0.0


def test_quartiles_outlier_detection() -> None:
    """Test quartiles for outlier detection."""
    # Data with outliers
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    result = quartiles(data)

    # 100 should be outside upper fence
    assert result["upper_fence"] < 100


# Test weighted_mean


def test_weighted_mean_equal_weights(simple_array: np.ndarray) -> None:
    """Test weighted_mean with equal weights."""
    weights = np.ones(len(simple_array))
    result = weighted_mean(simple_array, weights)

    assert result == pytest.approx(3.0)


def test_weighted_mean_no_weights(simple_array: np.ndarray) -> None:
    """Test weighted_mean without weights (should equal mean)."""
    result = weighted_mean(simple_array)

    assert result == pytest.approx(3.0)


def test_weighted_mean_linear_weights() -> None:
    """Test weighted_mean with linearly increasing weights."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = weighted_mean(data, weights)

    # Weighted mean should be higher than simple mean (3.67 vs 3.0)
    assert result > 3.0
    assert result == pytest.approx(np.sum(data * weights) / np.sum(weights))


def test_weighted_mean_with_trace(simple_trace: WaveformTrace) -> None:
    """Test weighted_mean with WaveformTrace."""
    weights = np.ones(len(simple_trace.data))
    result = weighted_mean(simple_trace, weights)

    assert result == pytest.approx(3.0)


def test_weighted_mean_single_value() -> None:
    """Test weighted_mean with single value."""
    data = np.array([42.0])
    weights = np.array([1.0])
    result = weighted_mean(data, weights)

    assert result == 42.0


# Test running_stats


def test_running_stats_basic(simple_array: np.ndarray) -> None:
    """Test running_stats with basic input."""
    result = running_stats(simple_array, window_size=3)

    assert "mean" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert len(result["mean"]) == len(simple_array) - 3 + 1


def test_running_stats_window_size_one() -> None:
    """Test running_stats with window size of 1."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = running_stats(data, window_size=1)

    np.testing.assert_array_equal(result["mean"], data)
    np.testing.assert_array_almost_equal(result["std"], np.zeros(len(data)))


def test_running_stats_window_equals_length() -> None:
    """Test running_stats with window size equal to data length."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = running_stats(data, window_size=len(data))

    assert len(result["mean"]) == 1
    assert result["mean"][0] == pytest.approx(3.0)


def test_running_stats_window_larger_than_data() -> None:
    """Test running_stats with window size larger than data."""
    data = np.array([1.0, 2.0, 3.0])
    result = running_stats(data, window_size=10)

    # Should cap window size to data length
    assert len(result["mean"]) == 1


def test_running_stats_with_trace(simple_trace: WaveformTrace) -> None:
    """Test running_stats with WaveformTrace."""
    result = running_stats(simple_trace, window_size=3)

    assert len(result["mean"]) > 0


def test_running_stats_correctness() -> None:
    """Test running_stats produces correct values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = running_stats(data, window_size=3)

    # First window: [1, 2, 3]
    assert result["mean"][0] == pytest.approx(2.0)
    assert result["min"][0] == 1.0
    assert result["max"][0] == 3.0

    # Last window: [3, 4, 5]
    assert result["mean"][-1] == pytest.approx(4.0)
    assert result["min"][-1] == 3.0
    assert result["max"][-1] == 5.0


# Test summary_stats


def test_summary_stats_comprehensive(simple_array: np.ndarray) -> None:
    """Test summary_stats includes all expected fields."""
    result = summary_stats(simple_array)

    # From basic_stats
    assert "mean" in result
    assert "variance" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert "range" in result
    assert "count" in result

    # From quartiles
    assert "q1" in result
    assert "median" in result
    assert "q3" in result
    assert "iqr" in result
    assert "lower_fence" in result
    assert "upper_fence" in result

    # Additional measures
    assert "median_abs_dev" in result
    assert "rms" in result
    assert "peak_to_rms" in result


def test_summary_stats_rms_calculation() -> None:
    """Test summary_stats RMS calculation."""
    data = np.array([3.0, 4.0])  # 3-4-5 triangle
    result = summary_stats(data)

    expected_rms = np.sqrt((9 + 16) / 2)
    assert result["rms"] == pytest.approx(expected_rms)


def test_summary_stats_mad_calculation() -> None:
    """Test summary_stats median absolute deviation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summary_stats(data)

    # Median is 3, MAD should be median of |data - 3|
    expected_mad = np.median(np.abs(data - 3.0))
    assert result["median_abs_dev"] == pytest.approx(expected_mad)


def test_summary_stats_with_trace(simple_trace: WaveformTrace) -> None:
    """Test summary_stats with WaveformTrace."""
    result = summary_stats(simple_trace)

    assert result["mean"] == pytest.approx(3.0)
    assert "rms" in result


def test_summary_stats_zero_rms() -> None:
    """Test summary_stats with zero RMS (uniform zero data)."""
    data = np.zeros(10)
    result = summary_stats(data)

    assert result["rms"] == 0.0
    assert np.isnan(result["peak_to_rms"])


def test_summary_stats_single_value() -> None:
    """Test summary_stats with single value."""
    data = np.array([42.0])
    result = summary_stats(data)

    assert result["mean"] == 42.0
    assert result["median"] == 42.0
    assert result["rms"] == 42.0


# Test weighted_std


def test_weighted_std_uniform_weights() -> None:
    """Test weighted_std with uniform weights equals unweighted std."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(len(data))

    wstd = weighted_std(data, weights)
    std = np.std(data)

    assert wstd == pytest.approx(std)


def test_weighted_std_no_weights() -> None:
    """Test weighted_std with None weights equals unweighted std."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    wstd = weighted_std(data, weights=None)
    std = np.std(data)

    assert wstd == pytest.approx(std)


def test_weighted_std_with_trace(simple_trace: WaveformTrace) -> None:
    """Test weighted_std with WaveformTrace."""
    weights = np.linspace(0.5, 1.0, len(simple_trace.data))
    wstd = weighted_std(simple_trace, weights)

    assert wstd > 0
    assert np.isfinite(wstd)


def test_weighted_std_ddof() -> None:
    """Test weighted_std with different ddof values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(len(data))

    wstd_ddof0 = weighted_std(data, weights, ddof=0)
    wstd_ddof1 = weighted_std(data, weights, ddof=1)

    # ddof=1 should give larger std (Bessel's correction)
    assert wstd_ddof1 > wstd_ddof0


def test_weighted_std_higher_weight_on_extremes() -> None:
    """Test weighted_std increases when extremes have higher weights."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Uniform weights
    uniform_weights = np.ones(len(data))
    wstd_uniform = weighted_std(data, uniform_weights)

    # Higher weights on extremes
    extreme_weights = np.array([2.0, 1.0, 1.0, 1.0, 2.0])
    wstd_extreme = weighted_std(data, extreme_weights)

    # Weighted std should be larger when extremes weighted more
    assert wstd_extreme > wstd_uniform


def test_weighted_std_known_values() -> None:
    """Test weighted_std with known analytical result."""
    # Simple case: two values with known weights
    data = np.array([1.0, 3.0])
    weights = np.array([1.0, 1.0])

    # Weighted mean = (1 + 3) / 2 = 2
    # Weighted variance = [(1-2)^2 + (3-2)^2] / 2 = [1 + 1] / 2 = 1
    # Weighted std = sqrt(1) = 1
    wstd = weighted_std(data, weights)
    assert wstd == pytest.approx(1.0)


def test_weighted_std_different_length_error() -> None:
    """Test weighted_std raises error when weights and data have different lengths."""
    data = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0])  # Wrong length

    with pytest.raises(ValueError, match="must have same length"):
        weighted_std(data, weights)


def test_weighted_std_negative_weights_error() -> None:
    """Test weighted_std raises error with negative weights."""
    data = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="must be non-negative"):
        weighted_std(data, weights)


def test_weighted_std_zero_weights() -> None:
    """Test weighted_std with all zero weights returns nan."""
    data = np.array([1.0, 2.0, 3.0])
    weights = np.zeros(len(data))

    wstd = weighted_std(data, weights)
    assert np.isnan(wstd)


def test_weighted_std_empty_array() -> None:
    """Test weighted_std with empty array returns nan."""
    data = np.array([])
    weights = np.array([])

    wstd = weighted_std(data, weights)
    assert np.isnan(wstd)


def test_weighted_std_single_value() -> None:
    """Test weighted_std with single value returns zero."""
    data = np.array([42.0])
    weights = np.array([1.0])

    wstd = weighted_std(data, weights)
    assert wstd == 0.0


def test_weighted_std_linear_weights() -> None:
    """Test weighted_std with linearly increasing weights."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    weights = np.linspace(0.1, 1.0, len(data))

    wstd = weighted_std(data, weights)

    assert wstd > 0
    assert np.isfinite(wstd)


def test_weighted_std_comparison_with_numpy() -> None:
    """Test weighted_std matches numpy for uniform weights."""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    weights = np.ones(len(data))

    wstd_ddof0 = weighted_std(data, weights, ddof=0)
    numpy_std = np.std(data, ddof=0)

    assert wstd_ddof0 == pytest.approx(numpy_std)


# Test measure function


def test_measure_default_with_units(simple_array: np.ndarray) -> None:
    """Test measure with default parameters returns all measurements with units."""
    result = measure(simple_array)

    # Check basic stats are present
    assert "mean" in result
    assert "variance" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert "range" in result
    assert "count" in result

    # Check percentiles are present
    assert "p1" in result
    assert "p5" in result
    assert "p25" in result
    assert "p50" in result
    assert "p75" in result
    assert "p95" in result
    assert "p99" in result

    # Check units structure
    assert "value" in result["mean"]
    assert "unit" in result["mean"]
    assert result["mean"]["unit"] == "V"
    assert result["std"]["unit"] == "V"
    assert result["variance"]["unit"] == "V²"
    assert result["count"]["unit"] == "samples"


def test_measure_with_trace(simple_trace: WaveformTrace) -> None:
    """Test measure with WaveformTrace input."""
    result = measure(simple_trace)

    # Should work with WaveformTrace and extract data
    assert "mean" in result
    assert result["mean"]["value"] == pytest.approx(3.0)
    assert result["mean"]["unit"] == "V"


def test_measure_without_units(simple_array: np.ndarray) -> None:
    """Test measure with include_units=False returns flat values."""
    result = measure(simple_array, include_units=False)

    # Check basic stats are present as flat values
    assert "mean" in result
    assert "std" in result
    assert "p50" in result

    # Check values are floats, not dicts
    assert isinstance(result["mean"], float)
    assert isinstance(result["std"], float)
    assert isinstance(result["p50"], float)

    # Verify actual values
    assert result["mean"] == pytest.approx(3.0)


def test_measure_with_specific_parameters(simple_array: np.ndarray) -> None:
    """Test measure with specific parameters list."""
    result = measure(simple_array, parameters=["mean", "std", "p50"])

    # Should only include requested parameters
    assert "mean" in result
    assert "std" in result
    assert "p50" in result

    # Should not include other measurements
    assert "variance" not in result
    assert "min" not in result
    assert "p1" not in result
    assert "p99" not in result

    # Check structure
    assert "value" in result["mean"]
    assert "unit" in result["mean"]


def test_measure_with_parameters_no_units(simple_array: np.ndarray) -> None:
    """Test measure with parameters and include_units=False."""
    result = measure(simple_array, parameters=["mean", "std"], include_units=False)

    # Should only include requested parameters as flat values
    assert len(result) == 2
    assert "mean" in result
    assert "std" in result
    assert isinstance(result["mean"], float)
    assert isinstance(result["std"], float)


def test_measure_empty_parameters_list() -> None:
    """Test measure with empty parameters list."""
    data = np.array([1.0, 2.0, 3.0])
    result = measure(data, parameters=[])

    # Should return empty dict
    assert len(result) == 0


def test_measure_invalid_parameters() -> None:
    """Test measure with invalid parameter names."""
    data = np.array([1.0, 2.0, 3.0])
    result = measure(data, parameters=["mean", "invalid_param", "std"])

    # Should only include valid parameters
    assert "mean" in result
    assert "std" in result
    assert "invalid_param" not in result


def test_measure_percentile_values(simple_array: np.ndarray) -> None:
    """Test measure includes correct percentile values."""
    result = measure(simple_array, include_units=False)

    # Verify percentile values are computed correctly
    assert result["p1"] == pytest.approx(np.percentile(simple_array, 1))
    assert result["p5"] == pytest.approx(np.percentile(simple_array, 5))
    assert result["p25"] == pytest.approx(np.percentile(simple_array, 25))
    assert result["p50"] == pytest.approx(np.percentile(simple_array, 50))
    assert result["p75"] == pytest.approx(np.percentile(simple_array, 75))
    assert result["p95"] == pytest.approx(np.percentile(simple_array, 95))
    assert result["p99"] == pytest.approx(np.percentile(simple_array, 99))


def test_measure_unit_mappings() -> None:
    """Test measure uses correct unit mappings."""
    data = np.array([1.0, 2.0, 3.0])
    result = measure(data)

    # Check specific unit mappings
    assert result["mean"]["unit"] == "V"
    assert result["variance"]["unit"] == "V²"
    assert result["std"]["unit"] == "V"
    assert result["min"]["unit"] == "V"
    assert result["max"]["unit"] == "V"
    assert result["range"]["unit"] == "dimensionless"
    assert result["count"]["unit"] == "samples"
    assert result["p1"]["unit"] == "dimensionless"
    assert result["p50"]["unit"] == "dimensionless"


def test_measure_with_gaussian_array(gaussian_array: np.ndarray) -> None:
    """Test measure with gaussian distributed data."""
    result = measure(gaussian_array, include_units=False)

    # Check basic statistical properties of gaussian data
    assert result["mean"] == pytest.approx(0.0, abs=0.1)
    assert result["std"] == pytest.approx(1.0, abs=0.1)

    # Percentiles should be symmetric around median
    assert abs(result["p50"]) < 0.1  # Median near 0
    assert result["p5"] < 0  # Lower percentile negative
    assert result["p95"] > 0  # Upper percentile positive


def test_measure_single_value() -> None:
    """Test measure with single value array."""
    data = np.array([42.0])
    result = measure(data, include_units=False)

    assert result["mean"] == 42.0
    assert result["std"] == 0.0
    assert result["min"] == 42.0
    assert result["max"] == 42.0
    assert result["p50"] == 42.0
