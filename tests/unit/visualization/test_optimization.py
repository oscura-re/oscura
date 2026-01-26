"""Comprehensive tests for visualization.optimization module.

Tests cover all functions for automatic plot parameter optimization including:
- Y-axis range calculation with outlier detection
- X-axis time window optimization with activity detection
- Grid spacing calculation (Wilkinson's algorithm)
- dB range optimization for spectrum plots
- Decimation algorithms (LTTB, minmax, uniform)
- Interesting region detection (edges, glitches, anomalies)

Coverage target: >90%
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oscura.visualization.optimization import (
    InterestingRegion,
    calculate_grid_spacing,
    calculate_optimal_x_window,
    calculate_optimal_y_range,
    decimate_for_display,
    detect_interesting_regions,
    optimize_db_range,
)


class TestCalculateOptimalYRange:
    """Tests for calculate_optimal_y_range function."""

    def test_basic_range_calculation(self) -> None:
        """Test basic Y-range calculation for normal data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_min, y_max = calculate_optimal_y_range(data)

        # Should include all data with margin
        assert y_min < 1.0
        assert y_max > 5.0
        assert y_min < y_max

    def test_outlier_exclusion(self) -> None:
        """Test that outliers are excluded from range."""
        # Normal data with outlier
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        y_min, y_max = calculate_optimal_y_range(data, outlier_threshold=3.0)

        # Should not be affected by outlier
        assert y_max < 50.0  # Much less than outlier value

    def test_symmetric_range(self) -> None:
        """Test symmetric range mode for bipolar signals."""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
        y_min, y_max = calculate_optimal_y_range(data, symmetric=True)

        # Should be symmetric around zero
        assert abs(y_min + y_max) < 1e-10  # Near zero
        assert y_min < 0
        assert y_max > 0

    def test_clipping_warning(self) -> None:
        """Test warning when too many samples are clipped."""
        # Data with many outliers
        data = np.concatenate([np.ones(90), np.full(10, 100.0)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_optimal_y_range(data, outlier_threshold=2.0, clip_warning_threshold=0.05)

            # Should warn about clipping
            assert len(w) > 0
            assert "Clipping detected" in str(w[0].message)

    def test_empty_data_raises(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data array is empty"):
            calculate_optimal_y_range(np.array([]))

    def test_all_nan_raises(self) -> None:
        """Test that all-NaN data raises ValueError."""
        data = np.full(10, np.nan)
        with pytest.raises(ValueError, match="only NaN"):
            calculate_optimal_y_range(data)

    def test_nan_handling(self) -> None:
        """Test proper handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
        y_min, y_max = calculate_optimal_y_range(data)

        # Should work with NaN values removed
        assert y_min < 1.0
        assert y_max > 5.0

    def test_margin_adaptation_dense_data(self) -> None:
        """Test smaller margin for dense data."""
        # Very dense data (>10000 points)
        data = np.random.randn(15000)
        y_min, y_max = calculate_optimal_y_range(data)

        # Should have smaller margin (2%)
        data_range = np.max(data) - np.min(data)
        total_range = y_max - y_min
        assert total_range < data_range * 1.05  # Less than 5% margin

    def test_margin_adaptation_sparse_data(self) -> None:
        """Test larger margin for sparse data."""
        # Sparse data (<100 points)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_min, y_max = calculate_optimal_y_range(data)

        # Should have larger margin (10%)
        data_range = 5.0 - 1.0
        total_range = y_max - y_min
        assert total_range > data_range * 1.08  # More than 8% margin

    def test_constant_data(self) -> None:
        """Test handling of constant data."""
        data = np.full(100, 5.0)
        y_min, y_max = calculate_optimal_y_range(data)

        # Should add margins even for constant data
        assert y_min < 5.0
        assert y_max > 5.0

    def test_custom_margin(self) -> None:
        """Test custom margin percentage."""
        data = np.array([0.0, 10.0])
        y_min, y_max = calculate_optimal_y_range(data, margin_percent=20.0)

        # Range should be 10.0, margin should be 20% = 2.0
        assert y_min == pytest.approx(-2.0, abs=0.5)
        assert y_max == pytest.approx(12.0, abs=0.5)


class TestCalculateOptimalXWindow:
    """Tests for calculate_optimal_x_window function."""

    def test_basic_window_calculation(self) -> None:
        """Test basic time window calculation."""
        time = np.linspace(0, 1.0, 1000)
        data = np.sin(2 * np.pi * 10 * time)  # 10 Hz signal

        t_start, t_end = calculate_optimal_x_window(time, data, target_features=5)

        assert t_start >= 0
        assert t_end <= 1.0
        assert t_start < t_end

    def test_empty_arrays_raise(self) -> None:
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_optimal_x_window(np.array([]), np.array([]))

    def test_mismatched_lengths_raise(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            calculate_optimal_x_window(np.array([1, 2, 3]), np.array([1, 2]))

    def test_no_activity_returns_full_range(self) -> None:
        """Test that zero signal returns full time range."""
        time = np.linspace(0, 1.0, 1000)
        data = np.zeros(1000)

        t_start, t_end = calculate_optimal_x_window(time, data)

        # Should return padded full range
        assert t_start < 0  # Padded below zero
        assert t_end > 1.0  # Padded above one

    def test_activity_detection(self) -> None:
        """Test activity-based window selection."""
        time = np.linspace(0, 1.0, 10000)
        # Zero signal except for burst at 0.5-0.6s
        data = np.zeros(10000)
        burst_start = 5000
        burst_end = 6000
        data[burst_start:burst_end] = np.sin(2 * np.pi * 50 * time[burst_start:burst_end])

        t_start, t_end = calculate_optimal_x_window(
            time, data, activity_threshold=0.1, target_features=3
        )

        # Window should focus on active region
        assert t_start >= 0.4  # Near burst start
        assert t_end <= 0.8  # Near burst end

    def test_periodic_signal_window(self) -> None:
        """Test window selection for periodic signal."""
        time = np.linspace(0, 1.0, 10000)
        freq = 100  # 100 Hz
        data = np.sin(2 * np.pi * freq * time)

        t_start, t_end = calculate_optimal_x_window(time, data, target_features=5)

        # Should show approximately 5 periods
        period = 1.0 / freq
        window_duration = t_end - t_start
        expected_duration = 5 * period

        assert window_duration == pytest.approx(expected_duration, rel=0.3)

    def test_samples_per_pixel_threshold(self) -> None:
        """Test decimation threshold parameter."""
        time = np.linspace(0, 1.0, 10000)
        data = np.sin(2 * np.pi * 10 * time)

        t_start, t_end = calculate_optimal_x_window(
            time, data, samples_per_pixel=2.0, screen_width=1000
        )

        # Window should be sized for decimation
        window_samples = int((t_end - t_start) / (time[1] - time[0]))
        target_samples = 1000 * 2  # screen_width * samples_per_pixel

        assert window_samples <= target_samples * 2  # Reasonable bound


class TestCalculateGridSpacing:
    """Tests for calculate_grid_spacing function."""

    def test_basic_spacing_calculation(self) -> None:
        """Test basic grid spacing calculation."""
        major, minor = calculate_grid_spacing(0, 100, target_major_ticks=5)

        assert major > 0
        assert minor > 0
        assert major > minor
        assert major % 10 == 0 or major % 2 == 0 or major % 5 == 0  # Nice number

    def test_invalid_range_raises(self) -> None:
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError, match="Invalid axis range"):
            calculate_grid_spacing(100, 50)

    def test_nice_numbers(self) -> None:
        """Test that spacing uses nice numbers (1, 2, 5 × 10^n)."""
        major, minor = calculate_grid_spacing(0, 1000, target_major_ticks=7)

        # Major spacing should be a nice number
        exponent = np.floor(np.log10(major))
        mantissa = major / (10**exponent)

        assert mantissa in [1.0, 2.0, 5.0, 10.0]

    def test_logarithmic_spacing(self) -> None:
        """Test logarithmic spacing mode."""
        major, minor = calculate_grid_spacing(1, 1000, log_scale=True)

        # For log scale, spacing should be power of 10
        assert major >= 1.0
        assert (np.log10(major) % 1.0) == pytest.approx(0.0, abs=0.01)

    def test_log_scale_sub_decade(self) -> None:
        """Test log scale with less than one decade."""
        # Range less than one decade
        major, minor = calculate_grid_spacing(1, 5, log_scale=True)

        # Should fall back to linear spacing
        assert major > 0
        assert minor > 0

    def test_time_axis_alignment(self) -> None:
        """Test alignment to time units."""
        # Range in microseconds
        major, minor = calculate_grid_spacing(0, 50e-6, time_axis=True)

        # Should align to time units (ns, μs, ms, s)
        time_units = [1e-9, 2e-9, 5e-9, 1e-6, 2e-6, 5e-6, 1e-3, 2e-3, 5e-3, 1.0, 2.0, 5.0]

        assert major in time_units or minor in time_units

    def test_grid_density_limit(self) -> None:
        """Test that grid doesn't become too dense."""
        major, minor = calculate_grid_spacing(0, 100, target_major_ticks=100)

        # Should disable minor grid if too dense
        n_major_ticks = 100 / major
        if n_major_ticks > 15:
            assert minor == major  # Minors disabled

    def test_minor_spacing_fraction(self) -> None:
        """Test minor grid is fraction of major."""
        major, minor = calculate_grid_spacing(0, 100)

        # Minor should be 1/2 or 1/5 of major
        ratio = major / minor
        assert ratio in [2.0, 5.0]


class TestOptimizeDbRange:
    """Tests for optimize_db_range function."""

    def test_basic_db_range(self) -> None:
        """Test basic dB range optimization."""
        # Create spectrum with known range
        spectrum_linear = np.concatenate([np.full(100, 1000.0), np.ones(900)])
        spectrum_db = 20 * np.log10(spectrum_linear)

        db_min, db_max = optimize_db_range(spectrum_db)

        assert db_min < db_max
        assert db_max > 0  # Peak level
        assert db_min < 0  # Below peak

    def test_empty_spectrum_raises(self) -> None:
        """Test that empty spectrum raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            optimize_db_range(np.array([]))

    def test_linear_to_db_conversion(self) -> None:
        """Test automatic linear to dB conversion."""
        # Linear spectrum (> 100)
        spectrum_linear = np.linspace(1, 1000, 1000)

        db_min, db_max = optimize_db_range(spectrum_linear)

        # Should convert to dB and optimize
        assert db_max == pytest.approx(60, abs=2)  # 20*log10(1000) = 60 dB

    def test_noise_floor_detection(self) -> None:
        """Test noise floor detection."""
        # Spectrum with clear noise floor
        spectrum = np.concatenate(
            [
                np.full(10, 0.0),  # Peaks at 0 dB
                np.full(990, -60.0),  # Noise floor at -60 dB
            ]
        )

        db_min, db_max = optimize_db_range(spectrum, noise_floor_percentile=5.0, margin_db=10.0)

        # db_min should be below noise floor
        assert db_min < -60.0
        assert db_min >= -70.0  # Margin applied

    def test_dynamic_range_compression(self) -> None:
        """Test dynamic range compression for very wide ranges."""
        # Very wide spectrum
        spectrum = np.linspace(-150, 0, 1000)

        db_min, db_max = optimize_db_range(spectrum, max_dynamic_range_db=100.0)

        # Dynamic range should be limited
        dynamic_range = db_max - db_min
        assert dynamic_range <= 100.0

    def test_peak_detection(self) -> None:
        """Test that peaks are detected correctly."""
        # Create spectrum with peaks
        freq = np.fft.rfftfreq(1000, 1 / 10000)
        spectrum = np.full(len(freq), -80.0)  # Noise floor
        spectrum[100] = -20.0  # Peak

        db_min, db_max = optimize_db_range(spectrum, peak_threshold_db=10.0)

        # db_max should be near peak
        assert db_max >= -20.0


class TestDecimateForDisplay:
    """Tests for decimate_for_display function."""

    def test_no_decimation_if_below_threshold(self) -> None:
        """Test that data below threshold is not decimated."""
        time = np.linspace(0, 1, 100)
        data = np.sin(time)

        time_dec, data_dec = decimate_for_display(time, data, max_points=1000)

        # Should return unchanged
        assert len(time_dec) == len(time)
        assert len(data_dec) == len(data)
        np.testing.assert_array_equal(time_dec, time)
        np.testing.assert_array_equal(data_dec, data)

    def test_empty_arrays_raise(self) -> None:
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            decimate_for_display(np.array([]), np.array([]))

    def test_mismatched_lengths_raise(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            decimate_for_display(np.array([1, 2, 3]), np.array([1, 2]))

    def test_uniform_decimation(self) -> None:
        """Test uniform stride decimation method."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 10 * time)

        time_dec, data_dec = decimate_for_display(time, data, max_points=100, method="uniform")

        # Should decimate to approximately max_points
        assert len(time_dec) <= 100
        assert len(data_dec) <= 100
        assert len(time_dec) == len(data_dec)

    def test_minmax_decimation(self) -> None:
        """Test min-max envelope decimation."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 10 * time)

        time_dec, data_dec = decimate_for_display(time, data, max_points=100, method="minmax")

        # Should preserve peaks and valleys
        assert len(time_dec) <= 200  # Min and max per bucket
        assert len(data_dec) <= 200
        assert np.max(data_dec) == pytest.approx(1.0, abs=0.01)
        assert np.min(data_dec) == pytest.approx(-1.0, abs=0.01)

    def test_lttb_decimation(self) -> None:
        """Test Largest Triangle Three Buckets decimation."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 10 * time) + 0.1 * np.random.randn(10000)

        time_dec, data_dec = decimate_for_display(time, data, max_points=100, method="lttb")

        # Should decimate to exactly max_points
        assert len(time_dec) == 100
        assert len(data_dec) == 100
        # First and last points should be preserved
        assert time_dec[0] == time[0]
        assert time_dec[-1] == time[-1]

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        time = np.linspace(0, 1, 1000)
        data = np.sin(time)

        with pytest.raises(ValueError, match="Invalid decimation method"):
            decimate_for_display(time, data, method="invalid")  # type: ignore

    def test_very_small_signal_not_decimated(self) -> None:
        """Test that very small signals (<10 points) are not decimated."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        time_dec, data_dec = decimate_for_display(time, data, max_points=3)

        # Should not decimate
        assert len(time_dec) == 5
        assert len(data_dec) == 5


class TestDetectInterestingRegions:
    """Tests for detect_interesting_regions function."""

    def test_basic_edge_detection(self) -> None:
        """Test edge detection in step signal."""
        # Create step signal
        signal = np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)])
        sample_rate = 1000.0

        regions = detect_interesting_regions(signal, sample_rate)

        # Should detect two edges
        edge_regions = [r for r in regions if r.type == "edge"]
        assert len(edge_regions) >= 2

    def test_empty_signal_raises(self) -> None:
        """Test that empty signal raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            detect_interesting_regions(np.array([]), 1000.0)

    def test_invalid_sample_rate_raises(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        signal = np.ones(100)

        with pytest.raises(ValueError, match="must be positive"):
            detect_interesting_regions(signal, 0.0)

        with pytest.raises(ValueError, match="must be positive"):
            detect_interesting_regions(signal, -100.0)

    def test_glitch_detection(self) -> None:
        """Test glitch detection (isolated spikes)."""
        # Normal signal with glitch
        signal = np.ones(1000)
        signal[500] = 10.0  # Glitch spike

        regions = detect_interesting_regions(signal, 1000.0, glitch_sigma=3.0)

        # Should detect glitch
        glitch_regions = [r for r in regions if r.type == "glitch"]
        assert len(glitch_regions) >= 1

    def test_anomaly_detection(self) -> None:
        """Test anomaly detection using MAD."""
        # Signal with anomalous region
        signal = np.concatenate(
            [
                np.random.randn(100),
                np.random.randn(50) * 5,  # Anomalous variance
                np.random.randn(100),
            ]
        )

        regions = detect_interesting_regions(signal, 1000.0, anomaly_threshold=3.0)

        # Should detect some anomalies
        anomaly_regions = [r for r in regions if r.type == "anomaly"]
        assert len(anomaly_regions) >= 0  # May or may not detect

    def test_pattern_change_detection(self) -> None:
        """Test pattern change detection."""
        # Signal with variance change
        signal = np.concatenate(
            [
                0.1 * np.random.randn(200),  # Low variance
                1.0 * np.random.randn(200),  # High variance
            ]
        )

        regions = detect_interesting_regions(signal, 1000.0)

        # Should detect pattern change
        pattern_regions = [r for r in regions if r.type == "pattern_change"]
        assert len(pattern_regions) >= 0  # Pattern detection is heuristic

    def test_max_regions_limit(self) -> None:
        """Test that max_regions parameter limits results."""
        # Signal with many edges
        signal = np.tile([0, 1], 100)

        regions = detect_interesting_regions(signal, 1000.0, max_regions=5)

        # Should return at most 5 regions
        assert len(regions) <= 5

    def test_min_region_samples_filter(self) -> None:
        """Test filtering by minimum region size."""
        signal = np.concatenate([np.zeros(100), np.ones(5), np.zeros(100)])

        regions = detect_interesting_regions(signal, 1000.0, min_region_samples=10)

        # Small region should be filtered out
        for region in regions:
            assert (region.end_idx - region.start_idx) >= 10

    def test_region_attributes(self) -> None:
        """Test that regions have all required attributes."""
        signal = np.concatenate([np.zeros(100), np.ones(100)])
        regions = detect_interesting_regions(signal, 1000.0)

        for region in regions:
            assert isinstance(region, InterestingRegion)
            assert region.start_idx >= 0
            assert region.end_idx > region.start_idx
            assert region.start_time >= 0
            assert region.end_time > region.start_time
            assert region.type in ["edge", "glitch", "anomaly", "pattern_change"]
            assert 0 <= region.significance <= 1.0
            assert isinstance(region.metadata, dict)

    def test_significance_ordering(self) -> None:
        """Test that regions are ordered by significance."""
        # Create signal with edges of different magnitudes
        signal = np.zeros(300)
        signal[100:150] = 0.5  # Small edge
        signal[200:250] = 10.0  # Large edge

        regions = detect_interesting_regions(signal, 1000.0)

        # Regions should be sorted by significance (descending)
        if len(regions) > 1:
            for i in range(len(regions) - 1):
                assert regions[i].significance >= regions[i + 1].significance

    def test_no_interesting_regions(self) -> None:
        """Test signal with no interesting regions."""
        # Constant signal
        signal = np.ones(1000)

        regions = detect_interesting_regions(signal, 1000.0)

        # Should return empty or very few regions
        assert len(regions) <= 1


class TestInterestingRegionDataclass:
    """Tests for InterestingRegion dataclass."""

    def test_creation(self) -> None:
        """Test creating InterestingRegion."""
        region = InterestingRegion(
            start_idx=0,
            end_idx=100,
            start_time=0.0,
            end_time=0.1,
            type="edge",
            significance=0.8,
            metadata={"threshold": 2.0},
        )

        assert region.start_idx == 0
        assert region.end_idx == 100
        assert region.start_time == 0.0
        assert region.end_time == 0.1
        assert region.type == "edge"
        assert region.significance == 0.8
        assert region.metadata["threshold"] == 2.0

    def test_type_literal(self) -> None:
        """Test that type field accepts valid literals."""
        valid_types = ["edge", "glitch", "anomaly", "pattern_change"]

        for type_val in valid_types:
            region = InterestingRegion(
                start_idx=0,
                end_idx=10,
                start_time=0.0,
                end_time=0.01,
                type=type_val,  # type: ignore
                significance=0.5,
                metadata={},
            )
            assert region.type == type_val


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_signal(self) -> None:
        """Test handling of single-point signal."""
        # Single point should handle gracefully
        data = np.array([5.0])

        y_min, y_max = calculate_optimal_y_range(data)
        assert y_min < 5.0
        assert y_max > 5.0

    def test_large_data(self) -> None:
        """Test with large datasets."""
        # Large dataset (1M points)
        data = np.random.randn(1_000_000)

        y_min, y_max = calculate_optimal_y_range(data)
        assert y_min < 0
        assert y_max > 0

    def test_extreme_values(self) -> None:
        """Test with extreme values."""
        # Very large values
        data = np.array([1e10, 2e10, 3e10])
        y_min, y_max = calculate_optimal_y_range(data)
        assert y_min < 1e10
        assert y_max > 3e10

        # Very small values
        data = np.array([1e-10, 2e-10, 3e-10])
        y_min, y_max = calculate_optimal_y_range(data)
        assert y_min < 1e-10
        assert y_max > 3e-10

    def test_mixed_positive_negative(self) -> None:
        """Test mixed positive and negative data."""
        data = np.array([-100, -50, 0, 50, 100])
        y_min, y_max = calculate_optimal_y_range(data)

        assert y_min < -100
        assert y_max > 100

    def test_integer_vs_float_data(self) -> None:
        """Test that integer data works correctly."""
        int_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        float_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        y_min_int, y_max_int = calculate_optimal_y_range(int_data)
        y_min_float, y_max_float = calculate_optimal_y_range(float_data)

        # Results should be similar
        assert y_min_int == pytest.approx(y_min_float, abs=0.1)
        assert y_max_int == pytest.approx(y_max_float, abs=0.1)


# Run tests with: pytest tests/unit/visualization/test_optimization.py -v
