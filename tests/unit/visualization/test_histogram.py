"""Tests for visualization histogram utilities."""

import numpy as np
import pytest

from oscura.visualization.histogram import (
    _auto_select_method,
    _freedman_diaconis_bins,
    _scott_bins,
    _sturges_bins,
    calculate_bin_edges,
    calculate_optimal_bins,
)


class TestCalculateOptimalBins:
    """Test calculate_optimal_bins function."""

    def test_sturges_method(self) -> None:
        """Test Sturges' rule for bin calculation."""
        data = np.random.randn(1000)
        bins = calculate_optimal_bins(data, method="sturges")
        assert isinstance(bins, int)
        assert bins >= 5
        # Sturges: k = ceil(log2(1000) + 1) ≈ 11
        assert 9 <= bins <= 13

    def test_freedman_diaconis_method(self) -> None:
        """Test Freedman-Diaconis rule for bin calculation."""
        data = np.random.randn(1000)
        bins = calculate_optimal_bins(data, method="freedman-diaconis")
        assert isinstance(bins, int)
        assert bins >= 5

    def test_scott_method(self) -> None:
        """Test Scott's rule for bin calculation."""
        data = np.random.randn(1000)
        bins = calculate_optimal_bins(data, method="scott")
        assert isinstance(bins, int)
        assert bins >= 5

    def test_auto_method(self) -> None:
        """Test automatic method selection."""
        data = np.random.randn(1000)
        bins = calculate_optimal_bins(data, method="auto")
        assert isinstance(bins, int)
        assert bins >= 5

    def test_min_bins_constraint(self) -> None:
        """Test minimum bins constraint."""
        data = np.array([1.0, 2.0, 3.0])  # Very small dataset
        bins = calculate_optimal_bins(data, min_bins=10)
        assert bins >= 10

    def test_max_bins_constraint(self) -> None:
        """Test maximum bins constraint."""
        data = np.random.randn(100000)  # Very large dataset
        bins = calculate_optimal_bins(data, max_bins=50)
        assert bins <= 50

    def test_min_max_bins_range(self) -> None:
        """Test bins within specified range."""
        data = np.random.randn(1000)
        bins = calculate_optimal_bins(data, min_bins=15, max_bins=25)
        assert 15 <= bins <= 25

    def test_empty_data_error(self) -> None:
        """Test error handling for empty data."""
        data = np.array([])
        with pytest.raises(ValueError, match="Data array cannot be empty"):
            calculate_optimal_bins(data)

    def test_invalid_min_bins_error(self) -> None:
        """Test error handling for invalid min_bins."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="min_bins must be >= 1"):
            calculate_optimal_bins(data, min_bins=0)

    def test_invalid_max_bins_error(self) -> None:
        """Test error handling for invalid max_bins."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="max_bins must be >= min_bins"):
            calculate_optimal_bins(data, min_bins=20, max_bins=10)

    def test_unknown_method_error(self) -> None:
        """Test error handling for unknown method."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_optimal_bins(data, method="invalid")  # type: ignore[arg-type]

    def test_data_with_nans(self) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
        bins = calculate_optimal_bins(data)
        assert isinstance(bins, int)
        assert bins >= 5

    def test_single_value_after_nan_removal(self) -> None:
        """Test handling when only one value remains after NaN removal."""
        data = np.array([1.0, np.nan, np.nan, np.nan])
        bins = calculate_optimal_bins(data, min_bins=5)
        assert bins == 5

    def test_small_dataset(self) -> None:
        """Test with very small dataset."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_large_dataset(self) -> None:
        """Test with large dataset."""
        data = np.random.randn(100000)
        bins = calculate_optimal_bins(data)
        assert 5 <= bins <= 200

    def test_uniform_distribution(self) -> None:
        """Test with uniform distribution."""
        data = np.random.uniform(0, 10, 1000)
        bins = calculate_optimal_bins(data)
        assert isinstance(bins, int)
        assert bins >= 5

    def test_skewed_distribution(self) -> None:
        """Test with skewed distribution."""
        data = np.random.exponential(2.0, 1000)
        bins = calculate_optimal_bins(data, method="auto")
        # Auto should select freedman-diaconis for skewed data
        assert isinstance(bins, int)


class TestCalculateBinEdges:
    """Test calculate_bin_edges function."""

    def test_basic_bin_edges(self) -> None:
        """Test basic bin edge calculation."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        edges = calculate_bin_edges(data, n_bins=5)
        assert len(edges) == 6  # n_bins + 1
        assert edges[0] == 0.0
        assert edges[-1] == 5.0
        # Edges should be evenly spaced
        spacing = np.diff(edges)
        assert np.allclose(spacing, spacing[0])

    def test_bin_edges_with_nans(self) -> None:
        """Test bin edge calculation with NaN values."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        edges = calculate_bin_edges(data, n_bins=3)
        assert len(edges) == 4
        assert not np.any(np.isnan(edges))

    def test_all_nans(self) -> None:
        """Test with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        edges = calculate_bin_edges(data, n_bins=5)
        assert len(edges) == 2
        assert edges[0] == 0.0
        assert edges[1] == 1.0

    def test_single_value_data(self) -> None:
        """Test with single unique value."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        edges = calculate_bin_edges(data, n_bins=10)
        assert len(edges) == 2
        assert edges[0] == 4.5
        assert edges[1] == 5.5

    def test_empty_data_error(self) -> None:
        """Test error handling for empty data."""
        data = np.array([])
        with pytest.raises(ValueError, match="Data array cannot be empty"):
            calculate_bin_edges(data, n_bins=5)

    def test_invalid_n_bins_error(self) -> None:
        """Test error handling for invalid n_bins."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            calculate_bin_edges(data, n_bins=0)

    def test_single_bin(self) -> None:
        """Test with single bin."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = calculate_bin_edges(data, n_bins=1)
        assert len(edges) == 2
        assert edges[0] == 1.0
        assert edges[1] == 5.0

    def test_many_bins(self) -> None:
        """Test with many bins."""
        data = np.random.randn(1000)
        edges = calculate_bin_edges(data, n_bins=100)
        assert len(edges) == 101
        # Edges should be monotonically increasing
        assert np.all(np.diff(edges) > 0)


class TestSturgesBins:
    """Test _sturges_bins function."""

    def test_sturges_small_sample(self) -> None:
        """Test Sturges' rule for small sample."""
        data = np.random.randn(10)
        bins = _sturges_bins(data)
        # log2(10) + 1 ≈ 4.32, ceil -> 5
        assert bins == 5

    def test_sturges_medium_sample(self) -> None:
        """Test Sturges' rule for medium sample."""
        data = np.random.randn(100)
        bins = _sturges_bins(data)
        # log2(100) + 1 ≈ 7.64, ceil -> 8
        assert bins == 8

    def test_sturges_large_sample(self) -> None:
        """Test Sturges' rule for large sample."""
        data = np.random.randn(1000)
        bins = _sturges_bins(data)
        # log2(1000) + 1 ≈ 10.97, ceil -> 11
        assert bins == 11

    def test_sturges_very_small_sample(self) -> None:
        """Test Sturges' rule for very small sample."""
        data = np.array([1.0, 2.0])
        bins = _sturges_bins(data)
        # log2(2) + 1 = 2
        assert bins == 2


class TestFreedmanDiaconisBins:
    """Test _freedman_diaconis_bins function."""

    def test_freedman_diaconis_normal(self) -> None:
        """Test Freedman-Diaconis rule for normal distribution."""
        np.random.seed(42)
        data = np.random.randn(1000)
        bins = _freedman_diaconis_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1

    def test_freedman_diaconis_uniform(self) -> None:
        """Test Freedman-Diaconis rule for uniform distribution."""
        data = np.random.uniform(0, 10, 1000)
        bins = _freedman_diaconis_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1

    def test_freedman_diaconis_zero_iqr(self) -> None:
        """Test fallback when IQR is zero."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        bins = _freedman_diaconis_bins(data)
        # Should fall back to Sturges
        assert bins == _sturges_bins(data)

    def test_freedman_diaconis_with_outliers(self) -> None:
        """Test Freedman-Diaconis with outliers."""
        # Normal data plus outliers
        normal_data = np.random.randn(100)
        outliers = np.array([100.0, -100.0])
        data = np.concatenate([normal_data, outliers])
        bins = _freedman_diaconis_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1

    def test_freedman_diaconis_small_sample(self) -> None:
        """Test Freedman-Diaconis with small sample."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = _freedman_diaconis_bins(data)
        assert bins >= 1


class TestScottBins:
    """Test _scott_bins function."""

    def test_scott_normal(self) -> None:
        """Test Scott's rule for normal distribution."""
        np.random.seed(42)
        data = np.random.randn(1000)
        bins = _scott_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1

    def test_scott_uniform(self) -> None:
        """Test Scott's rule for uniform distribution."""
        data = np.random.uniform(0, 10, 1000)
        bins = _scott_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1

    def test_scott_zero_std(self) -> None:
        """Test fallback when std is zero."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        bins = _scott_bins(data)
        # Should fall back to Sturges
        assert bins == _sturges_bins(data)

    def test_scott_small_sample(self) -> None:
        """Test Scott's rule with small sample."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = _scott_bins(data)
        assert bins >= 1

    def test_scott_large_sample(self) -> None:
        """Test Scott's rule with large sample."""
        data = np.random.randn(10000)
        bins = _scott_bins(data)
        assert isinstance(bins, int)
        assert bins >= 1


class TestAutoSelectMethod:
    """Test _auto_select_method function."""

    def test_auto_select_small_sample(self) -> None:
        """Test auto-selection for small sample."""
        data = np.random.randn(50)
        method = _auto_select_method(data)
        assert method == "sturges"

    def test_auto_select_normal_distribution(self) -> None:
        """Test auto-selection for normal distribution."""
        np.random.seed(42)
        data = np.random.randn(1000)
        method = _auto_select_method(data)
        # Low skewness should select scott
        assert method in ["scott", "freedman-diaconis"]

    def test_auto_select_skewed_distribution(self) -> None:
        """Test auto-selection for skewed distribution."""
        np.random.seed(42)
        data = np.random.exponential(2.0, 1000)
        method = _auto_select_method(data)
        # High skewness should select freedman-diaconis
        assert method == "freedman-diaconis"

    def test_auto_select_zero_std(self) -> None:
        """Test auto-selection when std is zero."""
        data = np.array([5.0] * 200)
        method = _auto_select_method(data)
        # Should fall back to sturges
        assert method == "sturges"

    def test_auto_select_boundary_100(self) -> None:
        """Test boundary case with n=100."""
        np.random.seed(42)
        data = np.random.randn(100)
        method = _auto_select_method(data)
        # With 100 samples, boundary behavior - accept either
        assert method in ["sturges", "scott", "freedman-diaconis"]

    def test_auto_select_just_over_100(self) -> None:
        """Test just over 100 samples."""
        data = np.random.randn(101)
        method = _auto_select_method(data)
        # Should not use Sturges (unless skewed)
        assert method in ["scott", "freedman-diaconis"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_data(self) -> None:
        """Test with all negative data."""
        data = np.random.randn(1000) - 10.0
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_large_range(self) -> None:
        """Test with very large data range."""
        data = np.array([0.0, 1e10])
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_small_range(self) -> None:
        """Test with very small data range."""
        data = np.array([1.0, 1.0001, 1.0002, 1.0003])
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_integer_data(self) -> None:
        """Test with integer data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_bimodal_distribution(self) -> None:
        """Test with bimodal distribution."""
        data1 = np.random.randn(500)
        data2 = np.random.randn(500) + 10.0
        data = np.concatenate([data1, data2])
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_all_same_value(self) -> None:
        """Test with all same values."""
        data = np.array([42.0] * 1000)
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_two_distinct_values(self) -> None:
        """Test with two distinct values."""
        data = np.array([1.0] * 500 + [2.0] * 500)
        bins = calculate_optimal_bins(data)
        assert bins >= 5

    def test_reproducibility(self) -> None:
        """Test that results are reproducible."""
        np.random.seed(42)
        data = np.random.randn(1000)
        bins1 = calculate_optimal_bins(data, method="sturges")

        np.random.seed(42)
        data = np.random.randn(1000)
        bins2 = calculate_optimal_bins(data, method="sturges")

        assert bins1 == bins2
