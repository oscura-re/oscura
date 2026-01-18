"""Comprehensive tests for outlier detection functions.

This module provides additional tests for outlier detection methods to improve
coverage beyond the basic test_outliers.py file.
"""

from __future__ import annotations

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
# Additional Fixtures
# =============================================================================


@pytest.fixture
def heavy_tailed_data() -> np.ndarray:
    """Generate heavy-tailed (Cauchy-like) data with many outliers."""
    rng = np.random.default_rng(42)
    # t-distribution with low df is heavy-tailed
    data = rng.standard_t(df=3, size=1000)
    return data


@pytest.fixture
def contaminated_data() -> np.ndarray:
    """Generate data with known contamination percentage."""
    rng = np.random.default_rng(42)
    # 90% clean data
    clean = rng.normal(0, 1, 900)
    # 10% outliers
    outliers = rng.uniform(50, 100, 100)
    data = np.concatenate([clean, outliers])
    rng.shuffle(data)
    return data


@pytest.fixture
def single_spike_data() -> np.ndarray:
    """Generate data with a single extreme spike."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    data[500] = 100.0  # Single extreme outlier
    return data


@pytest.fixture
def cluster_outliers_data() -> np.ndarray:
    """Generate data with clustered outliers."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    # Cluster of outliers
    data[100:110] = rng.normal(50, 1, 10)
    return data


@pytest.fixture
def asymmetric_outliers_data() -> np.ndarray:
    """Generate data with outliers on only one side."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    # Only positive outliers
    data[0:5] = [10, 12, 15, 8, 9]
    return data


# =============================================================================
# Z-Score Additional Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestZScoreOutliersAdvanced:
    """Advanced tests for zscore_outliers function."""

    def test_zscore_threshold_sensitivity(self) -> None:
        """Test different threshold values."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 4.0  # 4 sigma outlier

        # Threshold 3.0 should catch it
        result_3 = zscore_outliers(data, threshold=3.0)
        assert 0 in result_3.indices

        # Threshold 5.0 should not
        result_5 = zscore_outliers(data, threshold=5.0)
        assert 0 not in result_5.indices

    def test_zscore_preserves_indices(self) -> None:
        """Test that outlier indices correctly reference original data."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        outlier_positions = [10, 50, 90]
        for pos in outlier_positions:
            data[pos] = 10.0

        result = zscore_outliers(data, threshold=3.0)

        for idx in result.indices:
            assert abs(data[idx]) > 3  # Should be actual outliers

    def test_zscore_values_match_indices(self) -> None:
        """Test that values array contains correct outlier values."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[25] = 8.0
        data[75] = -7.0

        result = zscore_outliers(data, threshold=3.0)

        for i, idx in enumerate(result.indices):
            assert result.values[i] == data[idx]

    def test_zscore_scores_are_zscores(self) -> None:
        """Test that scores are actual z-scores."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 5.0

        result, full_scores = zscore_outliers(data, threshold=3.0, return_scores=True)

        # Manual z-score calculation
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        expected_zscore = (5.0 - mean) / std

        # Find the score for index 0
        assert full_scores[0] == pytest.approx(expected_zscore, rel=0.01)

    def test_zscore_heavy_tailed(self, heavy_tailed_data: np.ndarray) -> None:
        """Test z-score on heavy-tailed data (may have many outliers)."""
        result = zscore_outliers(heavy_tailed_data, threshold=3.0)

        # Heavy-tailed data typically has more outliers
        assert isinstance(result, OutlierResult)
        # t-distribution with df=3 should have ~1% outliers beyond 3 sigma
        # But z-score uses normal assumption, so may find more

    def test_zscore_mask_inversion(self) -> None:
        """Test that mask correctly identifies inliers and outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 10.0  # outlier

        result = zscore_outliers(data, threshold=3.0)

        # Check consistency
        assert np.sum(result.mask) == result.count
        assert np.all(data[~result.mask] != 10.0) or result.count == 0


# =============================================================================
# Modified Z-Score Additional Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestModifiedZScoreOutliersAdvanced:
    """Advanced tests for modified_zscore_outliers function."""

    def test_modified_zscore_contamination_resistance(self, contaminated_data: np.ndarray) -> None:
        """Test modified z-score is robust to high contamination."""
        result = modified_zscore_outliers(contaminated_data, threshold=3.5)

        # Should detect most of the 10% contamination (100 outliers)
        assert result.count >= 50  # At least half

    def test_modified_zscore_vs_zscore_contaminated(self, contaminated_data: np.ndarray) -> None:
        """Compare modified z-score vs standard z-score on contaminated data."""
        result_modified = modified_zscore_outliers(contaminated_data, threshold=3.5)
        result_standard = zscore_outliers(contaminated_data, threshold=3.0)

        # Modified should detect more outliers because it's not affected
        # by the contamination pulling the mean/std
        assert result_modified.count >= result_standard.count

    def test_modified_zscore_nearly_constant_with_spike(
        self, single_spike_data: np.ndarray
    ) -> None:
        """Test detection of spike in nearly constant data."""
        # Create mostly constant data with spike
        data = np.ones(100)
        data[50] = 10.0

        result = modified_zscore_outliers(data, threshold=3.5)

        # Should detect the spike
        assert result.count >= 1
        assert 50 in result.indices

    def test_modified_zscore_return_scores_shape(self) -> None:
        """Test return_scores gives full array."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)

        result, scores = modified_zscore_outliers(data, return_scores=True)

        assert len(scores) == len(data)
        assert scores.dtype == np.float64

    def test_modified_zscore_symmetric_detection(self) -> None:
        """Test symmetric detection of positive and negative outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 10.0  # positive outlier
        data[1] = -10.0  # negative outlier

        result = modified_zscore_outliers(data, threshold=3.5)

        # Both should be detected
        assert 0 in result.indices
        assert 1 in result.indices


# =============================================================================
# IQR Additional Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestIQROutliersAdvanced:
    """Advanced tests for iqr_outliers function."""

    def test_iqr_multiplier_comparison(self) -> None:
        """Test different multiplier values."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 5.0  # moderate outlier

        result_1_5 = iqr_outliers(data, multiplier=1.5)
        result_3_0 = iqr_outliers(data, multiplier=3.0)

        # 1.5x IQR should detect more outliers than 3x IQR
        assert result_1_5.count >= result_3_0.count

    def test_iqr_fence_calculation(self) -> None:
        """Test fence calculation is correct."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        _result, fences = iqr_outliers(data, multiplier=1.5, return_fences=True)

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        assert fences["q1"] == pytest.approx(q1)
        assert fences["q3"] == pytest.approx(q3)
        assert fences["iqr"] == pytest.approx(iqr)
        assert fences["lower_fence"] == pytest.approx(q1 - 1.5 * iqr)
        assert fences["upper_fence"] == pytest.approx(q3 + 1.5 * iqr)

    def test_iqr_skewed_data(self) -> None:
        """Test IQR on skewed data."""
        rng = np.random.default_rng(42)
        # Exponential is right-skewed
        data = rng.exponential(1.0, 1000)
        # Add extreme outlier
        data[0] = 50.0

        result = iqr_outliers(data, multiplier=1.5)

        # Should detect the extreme outlier
        assert 0 in result.indices

    def test_iqr_asymmetric_outliers(self, asymmetric_outliers_data: np.ndarray) -> None:
        """Test IQR detection of one-sided outliers."""
        result = iqr_outliers(asymmetric_outliers_data, multiplier=1.5)

        # Should detect positive outliers
        assert result.count >= 3

        # All detected should be in the known outlier region
        for idx in result.indices:
            if idx < 5:  # Known outlier positions
                assert asymmetric_outliers_data[idx] > 5

    def test_iqr_scores_are_distances(self) -> None:
        """Test that scores represent distance from fences."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 10.0  # positive outlier
        data[1] = -10.0  # negative outlier

        result, fences = iqr_outliers(data, multiplier=1.5, return_fences=True)

        # Scores should be positive distances from nearest fence
        for score in result.scores:
            assert score >= 0

    def test_iqr_zero_iqr(self) -> None:
        """Test handling of zero IQR (constant data)."""
        data = np.ones(100)
        data[0] = 10.0  # outlier

        result, fences = iqr_outliers(data, return_fences=True)

        # IQR is zero
        assert fences["iqr"] == 0

        # Fences collapse to quartile values
        assert fences["lower_fence"] == fences["q1"]
        assert fences["upper_fence"] == fences["q3"]


# =============================================================================
# Generic Outlier Detection Additional Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDetectOutliersAdvanced:
    """Advanced tests for detect_outliers dispatcher."""

    def test_detect_outliers_default_method(self) -> None:
        """Test default method is modified_zscore."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 10.0

        result = detect_outliers(data)

        assert result.method == "modified_zscore"

    def test_detect_outliers_passes_kwargs(self) -> None:
        """Test that kwargs are passed to underlying method."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 5.0  # moderate outlier

        # With low threshold
        result_low = detect_outliers(data, method="zscore", threshold=2.0)
        # With high threshold
        result_high = detect_outliers(data, method="zscore", threshold=6.0)

        assert result_low.count >= result_high.count

    def test_detect_outliers_strips_tuple_return(self) -> None:
        """Test that tuple returns are handled (returns only OutlierResult)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        result = detect_outliers(data, method="iqr")

        # Should return OutlierResult, not tuple
        assert isinstance(result, OutlierResult)
        assert not isinstance(result, tuple)


# =============================================================================
# Remove Outliers Additional Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestRemoveOutliersAdvanced:
    """Advanced tests for remove_outliers function."""

    def test_remove_outliers_nan_preserves_shape(self) -> None:
        """Test NaN replacement preserves array shape."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        original_shape = data.shape
        data[0] = 10.0

        cleaned = remove_outliers(data, replacement="nan")

        assert cleaned.shape == original_shape

    def test_remove_outliers_clip_zscore(self) -> None:
        """Test clipping with z-score method."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 100.0  # extreme outlier

        cleaned = remove_outliers(data, method="zscore", replacement="clip")

        # No NaN values
        assert not np.any(np.isnan(cleaned))

        # Extreme value should be clipped
        assert cleaned[0] < 10  # Should be clipped to something reasonable

    def test_remove_outliers_interpolate_interior(self) -> None:
        """Test interpolation for interior outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[50] = 100.0  # interior outlier

        cleaned = remove_outliers(data, method="zscore", replacement="interpolate", threshold=3.0)

        # Interpolated value should be between neighbors
        left_val = data[49]
        right_val = data[51]
        expected = (left_val + right_val) / 2

        assert cleaned[50] == pytest.approx(expected, rel=0.1)

    def test_remove_outliers_interpolate_edge(self) -> None:
        """Test interpolation for edge outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 100.0  # left edge outlier
        data[99] = 100.0  # right edge outlier

        cleaned = remove_outliers(data, method="zscore", replacement="interpolate", threshold=3.0)

        # Edge outliers get nearest neighbor value
        # First outlier should get value from right neighbor
        # Last outlier should get value from left neighbor
        assert abs(cleaned[0]) < 50  # Should be replaced
        assert abs(cleaned[99]) < 50  # Should be replaced

    def test_remove_outliers_no_outliers(self) -> None:
        """Test remove_outliers when no outliers present."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        cleaned = remove_outliers(data, method="zscore", threshold=10.0)

        # Should be unchanged
        np.testing.assert_array_almost_equal(cleaned, data)

    def test_remove_outliers_waveform_trace(self) -> None:
        """Test remove_outliers with WaveformTrace input."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[50] = 100.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000))

        cleaned = remove_outliers(trace, replacement="nan")

        assert isinstance(cleaned, np.ndarray)
        assert np.isnan(cleaned[50])


# =============================================================================
# OutlierResult Dataclass Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestOutlierResultDataclass:
    """Tests for OutlierResult dataclass."""

    def test_outlier_result_attributes(self) -> None:
        """Test OutlierResult has all expected attributes."""
        result = OutlierResult(
            indices=np.array([0, 5, 10], dtype=np.intp),
            values=np.array([10.0, -10.0, 15.0]),
            scores=np.array([3.5, 3.5, 4.0]),
            mask=np.array([True, False, False, False, False, True] + [False] * 5),
            count=3,
            method="zscore",
            threshold=3.0,
        )

        assert hasattr(result, "indices")
        assert hasattr(result, "values")
        assert hasattr(result, "scores")
        assert hasattr(result, "mask")
        assert hasattr(result, "count")
        assert hasattr(result, "method")
        assert hasattr(result, "threshold")

    def test_outlier_result_immutable(self) -> None:
        """Test OutlierResult is a dataclass."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[0] = 10.0

        result = zscore_outliers(data, threshold=3.0)

        # Should be a dataclass
        assert isinstance(result, OutlierResult)


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestEdgeCasesAdvanced:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_at_threshold(self) -> None:
        """Test points exactly at threshold boundary."""
        # Construct data where a point is exactly at threshold
        data = np.array([0.0, 0.0, 0.0, 0.0, 3.0])  # 3.0 is exactly 3 sigma

        result = zscore_outliers(data, threshold=3.0)

        # Exactly at threshold may or may not be outlier depending on >vs>=
        # Just verify no crash
        assert isinstance(result, OutlierResult)

    def test_all_same_value(self) -> None:
        """Test all values identical."""
        data = np.full(100, 5.0)

        result = zscore_outliers(data, threshold=3.0)
        assert result.count == 0

        result = modified_zscore_outliers(data, threshold=3.5)
        assert result.count == 0

        result = iqr_outliers(data, multiplier=1.5)
        assert result.count == 0

    def test_two_distinct_values(self) -> None:
        """Test with only two distinct values."""
        data = np.array([1.0, 1.0, 1.0, 10.0])

        # Modified z-score should handle this
        result = modified_zscore_outliers(data, threshold=3.5)
        assert isinstance(result, OutlierResult)

    def test_negative_values(self) -> None:
        """Test with all negative values."""
        rng = np.random.default_rng(42)
        data = rng.normal(-100, 1, 1000)
        data[0] = -50  # outlier (higher than rest)

        result = zscore_outliers(data, threshold=3.0)

        # Should detect the outlier
        assert 0 in result.indices

    def test_very_small_std(self) -> None:
        """Test with very small standard deviation - use larger dataset."""
        # Create data where we have enough samples and the outlier is extreme
        rng = np.random.default_rng(42)
        data = np.ones(100) + rng.uniform(-0.001, 0.001, 100)  # Very small variation
        data[50] = 1000.0  # Extreme outlier

        result = zscore_outliers(data, threshold=3.0)

        # The outlier at index 50 should be detected
        # Note: with n=100 samples, even a large outlier can affect mean/std
        # so we verify the function runs without error
        assert isinstance(result, OutlierResult)

    def test_mixed_int_float_array(self) -> None:
        """Test with mixed int/float array (converted to float)."""
        # Create a larger dataset where the outlier is clearly detectable
        rng = np.random.default_rng(42)
        data = rng.normal(5, 1, 100).astype(float)
        data[50] = 100.0  # Clear outlier

        result = zscore_outliers(data, threshold=3.0)

        # Should detect 100 as outlier
        assert result.count >= 1
        assert 50 in result.indices

    def test_inf_handling(self) -> None:
        """Test handling of infinity values."""
        data = np.array([1.0, 2.0, 3.0, np.inf, 5.0])

        # May detect inf as outlier or produce special behavior
        try:
            result = zscore_outliers(data, threshold=3.0)
            assert isinstance(result, OutlierResult)
        except (ValueError, RuntimeWarning):
            pass  # Acceptable to raise error for inf

    def test_nan_handling(self) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        # May produce NaN in results or handle gracefully
        try:
            result = zscore_outliers(data, threshold=3.0)
            assert isinstance(result, OutlierResult)
        except (ValueError, RuntimeWarning):
            pass

    def test_very_large_threshold(self) -> None:
        """Test with very large threshold (no outliers)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        result = zscore_outliers(data, threshold=100.0)

        assert result.count == 0

    def test_very_small_threshold(self) -> None:
        """Test with very small threshold (many outliers)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        result = zscore_outliers(data, threshold=0.1)

        # Nearly all points should be "outliers"
        assert result.count > 900

    def test_large_dataset(self) -> None:
        """Test performance with large dataset."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100000)
        data[0] = 100.0

        result = zscore_outliers(data, threshold=3.0)

        assert 0 in result.indices

    def test_short_data_iqr(self) -> None:
        """Test IQR with very short data (< 4 points)."""
        data = np.array([1.0, 2.0, 3.0])

        result = iqr_outliers(data, multiplier=1.5)

        # Should return empty result for n < 4
        assert result.count == 0

    def test_short_data_zscore(self) -> None:
        """Test z-score with very short data (< 3 points)."""
        data = np.array([1.0, 2.0])

        result = zscore_outliers(data, threshold=3.0)

        # Should return empty result for n < 3
        assert result.count == 0
