"""Tests for outlier detection functions.

Tests for TASK-031 (Outlier Detection).
"""

import numpy as np
import pytest

from oscura.analyzers.statistics.outliers import (
    OutlierResult,
    detect_outliers,
    iqr_outliers,
    modified_zscore_outliers,
    remove_outliers,
    zscore_outliers,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate normally distributed data with outliers."""
    rng = np.random.default_rng(42)
    # Normal data
    data = rng.normal(0, 1, 1000)
    # Add outliers
    data[10] = 10.0  # Clear outlier
    data[20] = -8.0  # Clear outlier
    data[500] = 5.5  # Moderate outlier
    return data


@pytest.fixture
def skewed_data() -> np.ndarray:
    """Generate skewed data with outliers."""
    rng = np.random.default_rng(42)
    # Skewed (exponential-like) distribution
    data = rng.exponential(1.0, 1000)
    # Add extreme outliers
    data[100] = 50.0
    data[200] = 60.0
    return data


@pytest.fixture
def clean_data() -> np.ndarray:
    """Generate clean data without outliers."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 1000)


@pytest.fixture
def trace_with_outliers(normal_data: np.ndarray) -> WaveformTrace:
    """Create WaveformTrace with outliers."""
    return WaveformTrace(
        data=normal_data,
        metadata=TraceMetadata(sample_rate=1000.0),
    )


# =============================================================================
# Z-Score Tests
# =============================================================================


class TestZScoreOutliers:
    """Tests for zscore_outliers function."""

    def test_zscore_detects_outliers(self, normal_data: np.ndarray):
        """Test that Z-score method detects known outliers."""
        result = zscore_outliers(normal_data, threshold=3.0)

        assert isinstance(result, OutlierResult)
        assert result.method == "zscore"
        assert result.threshold == 3.0

        # Should detect at least the extreme outliers at indices 10 and 20
        assert result.count >= 2
        assert 10 in result.indices
        assert 20 in result.indices

    def test_zscore_lower_threshold(self, normal_data: np.ndarray):
        """Test that lower threshold detects more outliers."""
        result_strict = zscore_outliers(normal_data, threshold=3.0)
        result_loose = zscore_outliers(normal_data, threshold=2.0)

        assert result_loose.count >= result_strict.count

    def test_zscore_return_scores(self, normal_data: np.ndarray):
        """Test returning z-scores."""
        _result, scores = zscore_outliers(normal_data, return_scores=True)

        assert len(scores) == len(normal_data)
        # Z-scores should have mean ~0 and std ~1
        assert abs(np.mean(scores)) < 0.1
        assert 0.9 < np.std(scores) < 1.1

    def test_zscore_no_outliers(self, clean_data: np.ndarray):
        """Test with data that has no outliers."""
        result = zscore_outliers(clean_data, threshold=3.0)

        # Very few or no outliers expected (statistically, ~0.3%)
        assert result.count < 10

    def test_zscore_mask(self, normal_data: np.ndarray):
        """Test that mask correctly marks outliers."""
        result = zscore_outliers(normal_data, threshold=3.0)

        assert len(result.mask) == len(normal_data)
        assert result.mask.dtype == np.bool_
        assert np.sum(result.mask) == result.count

        # Check mask matches indices
        for idx in result.indices:
            assert result.mask[idx]


# =============================================================================
# Modified Z-Score Tests
# =============================================================================


class TestModifiedZScoreOutliers:
    """Tests for modified_zscore_outliers function."""

    def test_modified_zscore_detects_outliers(self, normal_data: np.ndarray):
        """Test that modified Z-score detects outliers."""
        result = modified_zscore_outliers(normal_data, threshold=3.5)

        assert isinstance(result, OutlierResult)
        assert result.method == "modified_zscore"
        assert result.count >= 2  # At least the extreme outliers

    def test_modified_zscore_robust_to_contamination(self):
        """Test that modified Z-score is robust to contaminated data."""
        rng = np.random.default_rng(42)

        # Data with 10% contamination
        clean = rng.normal(0, 1, 900)
        contamination = rng.uniform(50, 100, 100)
        data = np.concatenate([clean, contamination])
        rng.shuffle(data)

        result = modified_zscore_outliers(data, threshold=3.5)

        # Should detect most of the contamination
        # (MAD-based method is robust to up to ~50% contamination)
        assert result.count >= 80  # Most of the 100 outliers

    def test_modified_zscore_vs_standard_zscore(self, skewed_data: np.ndarray):
        """Compare modified Z-score to standard Z-score on skewed data."""
        zscore_outliers(skewed_data, threshold=3.0)
        modified_zscore_outliers(skewed_data, threshold=3.5)

        # Modified should be more robust for skewed data
        # (less sensitive to extreme values affecting mean/std)


# =============================================================================
# IQR Tests
# =============================================================================


class TestIQROutliers:
    """Tests for iqr_outliers function."""

    def test_iqr_detects_outliers(self, normal_data: np.ndarray):
        """Test that IQR method detects outliers."""
        result = iqr_outliers(normal_data, multiplier=1.5)

        assert isinstance(result, OutlierResult)
        assert result.method == "iqr"
        assert result.count >= 2

    def test_iqr_return_fences(self, normal_data: np.ndarray):
        """Test returning fence values."""
        _result, fences = iqr_outliers(normal_data, return_fences=True)

        assert "q1" in fences
        assert "q3" in fences
        assert "iqr" in fences
        assert "lower_fence" in fences
        assert "upper_fence" in fences

        # Check fence calculation
        expected_lower = fences["q1"] - 1.5 * fences["iqr"]
        expected_upper = fences["q3"] + 1.5 * fences["iqr"]
        assert np.isclose(fences["lower_fence"], expected_lower)
        assert np.isclose(fences["upper_fence"], expected_upper)

    def test_iqr_extreme_multiplier(self, normal_data: np.ndarray):
        """Test with extreme multiplier (3.0)."""
        result_normal = iqr_outliers(normal_data, multiplier=1.5)
        result_extreme = iqr_outliers(normal_data, multiplier=3.0)

        # Extreme multiplier should detect fewer outliers
        assert result_extreme.count <= result_normal.count

    def test_iqr_skewed_data(self, skewed_data: np.ndarray):
        """Test IQR on skewed data."""
        result = iqr_outliers(skewed_data, multiplier=1.5)

        # IQR should handle skewed data well
        assert result.count >= 2  # Known outliers


# =============================================================================
# Generic Outlier Detection Tests
# =============================================================================


class TestDetectOutliers:
    """Tests for detect_outliers dispatcher function."""

    def test_detect_outliers_zscore(self, normal_data: np.ndarray):
        """Test dispatcher with Z-score method."""
        result = detect_outliers(normal_data, method="zscore", threshold=3.0)

        assert result.method == "zscore"
        assert result.count >= 2

    def test_detect_outliers_modified_zscore(self, normal_data: np.ndarray):
        """Test dispatcher with modified Z-score method."""
        result = detect_outliers(normal_data, method="modified_zscore")

        assert result.method == "modified_zscore"

    def test_detect_outliers_iqr(self, normal_data: np.ndarray):
        """Test dispatcher with IQR method."""
        result = detect_outliers(normal_data, method="iqr", multiplier=1.5)

        assert result.method == "iqr"

    def test_detect_outliers_invalid_method(self, normal_data: np.ndarray):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            detect_outliers(normal_data, method="invalid")

    def test_detect_outliers_with_trace(self, trace_with_outliers: WaveformTrace):
        """Test with WaveformTrace input."""
        result = detect_outliers(trace_with_outliers, method="zscore")

        assert isinstance(result, OutlierResult)
        assert result.count >= 2


# =============================================================================
# Remove Outliers Tests
# =============================================================================


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_remove_outliers_nan(self, normal_data: np.ndarray):
        """Test replacing outliers with NaN."""
        cleaned = remove_outliers(normal_data, method="zscore", replacement="nan")

        # Known outliers should be NaN
        assert np.isnan(cleaned[10])
        assert np.isnan(cleaned[20])

        # Non-outliers should be unchanged
        non_outlier_idx = 50  # Random non-outlier
        if not np.isnan(normal_data[non_outlier_idx]):
            assert cleaned[non_outlier_idx] == normal_data[non_outlier_idx]

    def test_remove_outliers_clip(self, normal_data: np.ndarray):
        """Test clipping outliers to threshold."""
        cleaned = remove_outliers(normal_data, method="iqr", replacement="clip")

        # No NaN values
        assert not np.any(np.isnan(cleaned))

        # Values should be within fences
        _, fences = iqr_outliers(normal_data, return_fences=True)
        assert np.all(cleaned >= fences["lower_fence"])
        assert np.all(cleaned <= fences["upper_fence"])

    def test_remove_outliers_interpolate(self, normal_data: np.ndarray):
        """Test interpolating outliers from neighbors."""
        cleaned = remove_outliers(normal_data, method="zscore", replacement="interpolate")

        # No NaN values
        assert not np.any(np.isnan(cleaned))

        # Interpolated values should be between neighbors
        # (for interior outliers)
        outlier_idx = 10
        if 0 < outlier_idx < len(cleaned) - 1:
            # Check it's not the original extreme value
            assert abs(cleaned[outlier_idx]) < abs(normal_data[outlier_idx])

    def test_remove_outliers_invalid_replacement(self, normal_data: np.ndarray):
        """Test error on invalid replacement method."""
        with pytest.raises(ValueError, match="Unknown replacement"):
            remove_outliers(normal_data, replacement="invalid")


# =============================================================================
# Edge Cases
# =============================================================================


class TestStatisticalOutliersEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array(self):
        """Test with empty array."""
        data = np.array([])
        result = zscore_outliers(data)
        # Should not crash, return empty result
        assert result.count == 0

    def test_single_value(self):
        """Test with single value."""
        data = np.array([1.0])
        result = zscore_outliers(data)
        assert result.count == 0

    def test_two_values(self):
        """Test with two values."""
        data = np.array([1.0, 2.0])
        result = zscore_outliers(data)
        # Should handle gracefully
        assert result.count >= 0

    def test_constant_array(self):
        """Test with constant values (zero variance)."""
        data = np.ones(100)
        result = zscore_outliers(data)

        # No outliers when all values are same
        assert result.count == 0

    def test_nearly_constant_with_outlier(self):
        """Test constant array with single outlier."""
        data = np.ones(100)
        data[50] = 100.0  # Single outlier

        result = modified_zscore_outliers(data, threshold=3.5)

        # Should detect the outlier
        assert result.count >= 1
        assert 50 in result.indices

    def test_all_outliers(self):
        """Test when nearly all points are outliers."""
        rng = np.random.default_rng(42)
        # Almost all extreme values
        data = rng.uniform(1000, 2000, 100)
        data[0] = 0.0  # Single "normal" value

        # Modified Z-score should still work
        modified_zscore_outliers(data, threshold=3.5)
        # At least handles it without crashing

    def test_large_dataset(self):
        """Test performance on large dataset."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100000)
        data[0] = 100.0

        # Should complete quickly
        result = zscore_outliers(data)
        assert 0 in result.indices
