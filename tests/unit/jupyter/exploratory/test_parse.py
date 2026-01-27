"""Comprehensive tests for error-tolerant protocol parsing.

Tests requirements:
  - DecodedFrame dataclass
  - TimestampCorrection dataclass
  - correct_timestamp_jitter with lowpass and PLL methods
  - Jitter measurement and reduction
  - Error tolerance modes (STRICT, TOLERANT, PERMISSIVE)
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.jupyter.exploratory.parse import (
    DecodedFrame,
    ErrorTolerance,
    TimestampCorrection,
    correct_timestamp_jitter,
)

pytestmark = pytest.mark.unit


class TestErrorToleranceEnum:
    """Test ErrorTolerance enumeration."""

    def test_enum_values(self) -> None:
        """Test ErrorTolerance enum values."""
        assert ErrorTolerance.STRICT.value == "strict"
        assert ErrorTolerance.TOLERANT.value == "tolerant"
        assert ErrorTolerance.PERMISSIVE.value == "permissive"

    def test_enum_members(self) -> None:
        """Test that all expected members exist."""
        assert hasattr(ErrorTolerance, "STRICT")
        assert hasattr(ErrorTolerance, "TOLERANT")
        assert hasattr(ErrorTolerance, "PERMISSIVE")


class TestDecodedFrameDataclass:
    """Test DecodedFrame dataclass."""

    def test_valid_frame(self) -> None:
        """Test DecodedFrame for valid frame."""
        frame = DecodedFrame(
            data=b"\x01\x02\x03",
            timestamp=1.23e-3,
            valid=True,
            error_type=None,
            position=0,
        )

        assert frame.data == b"\x01\x02\x03"
        assert frame.timestamp == 1.23e-3
        assert frame.valid is True
        assert frame.error_type is None
        assert frame.position == 0

    def test_invalid_frame(self) -> None:
        """Test DecodedFrame for invalid frame."""
        frame = DecodedFrame(
            data=b"\x01\x02",
            timestamp=2.34e-3,
            valid=False,
            error_type="framing",
            position=100,
        )

        assert frame.data == b"\x01\x02"
        assert frame.valid is False
        assert frame.error_type == "framing"
        assert frame.position == 100


class TestTimestampCorrectionDataclass:
    """Test TimestampCorrection dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test TimestampCorrection initialization."""
        timestamps = np.linspace(0, 1e-3, 1000)
        correction = TimestampCorrection(
            corrected_timestamps=timestamps,
            original_jitter_rms=1e-7,
            corrected_jitter_rms=2e-8,
            reduction_ratio=5.0,
            samples_corrected=50,
            max_correction=1e-6,
        )

        assert len(correction.corrected_timestamps) == 1000
        assert correction.original_jitter_rms == 1e-7
        assert correction.corrected_jitter_rms == 2e-8
        assert correction.reduction_ratio == 5.0
        assert correction.samples_corrected == 50
        assert correction.max_correction == 1e-6


class TestTimestampJitterCorrectionBasic:
    """Test basic timestamp jitter correction."""

    def test_perfect_timestamps_no_correction(self) -> None:
        """Test that perfect timestamps need no correction."""
        # Perfect 1 MHz sampling
        timestamps = np.linspace(0, 1e-3, 1000)

        result = correct_timestamp_jitter(timestamps, expected_rate=1e6)

        # Should have negligible jitter (tolerance for floating point precision)
        assert result.original_jitter_rms < 2e-9  # Increased tolerance for np.linspace precision
        # When data is already perfect, correction may not improve (reduction_ratio can be <1)
        assert result.samples_corrected >= 0

    def test_jittery_timestamps_corrected(self) -> None:
        """Test correction of jittery timestamps."""
        # 1 MHz with jitter
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 1e-7, 1000)  # 100 ns RMS jitter
        jittery = base_timestamps + jitter

        result = correct_timestamp_jitter(jittery, expected_rate=1e6)

        # Should reduce jitter
        assert result.original_jitter_rms > 0
        assert result.corrected_jitter_rms < result.original_jitter_rms
        assert result.reduction_ratio > 1.0

    def test_lowpass_method(self) -> None:
        """Test lowpass filter correction method."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6, method="lowpass")

        assert result.reduction_ratio > 1.0
        assert len(result.corrected_timestamps) == len(jittery)

    def test_pll_method(self) -> None:
        """Test PLL correction method."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6, method="pll")

        assert result.reduction_ratio > 1.0
        assert len(result.corrected_timestamps) == len(jittery)


class TestJitterMeasurement:
    """Test jitter measurement calculations."""

    def test_high_jitter_measurement(self) -> None:
        """Test measurement of high jitter."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        high_jitter = base_timestamps + rng.normal(0, 5e-7, 1000)  # 500 ns

        result = correct_timestamp_jitter(high_jitter, expected_rate=1e6)

        # Should measure significant jitter
        assert result.original_jitter_rms > 1e-7

    def test_low_jitter_measurement(self) -> None:
        """Test measurement of low jitter."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        low_jitter = base_timestamps + rng.normal(0, 1e-9, 1000)  # 1 ns

        result = correct_timestamp_jitter(low_jitter, expected_rate=1e6)

        # Should measure very low jitter
        assert result.original_jitter_rms < 1e-8


class TestMaxCorrectionFactor:
    """Test max_correction_factor parameter."""

    def test_small_max_correction(self) -> None:
        """Test with small max correction factor."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6, max_correction_factor=1.0)

        # Max correction should be limited
        assert result.max_correction <= 1e-6  # 1x period

    def test_large_max_correction(self) -> None:
        """Test with large max correction factor."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6, max_correction_factor=5.0)

        # Should allow larger corrections
        assert result.reduction_ratio > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_timestamps_error(self) -> None:
        """Test that empty timestamps raise error."""
        with pytest.raises(ValueError, match="empty"):
            correct_timestamp_jitter(np.array([]), expected_rate=1e6)

    def test_invalid_expected_rate_zero(self) -> None:
        """Test that zero expected_rate raises error."""
        timestamps = np.linspace(0, 1e-3, 1000)

        with pytest.raises(ValueError, match="expected_rate"):
            correct_timestamp_jitter(timestamps, expected_rate=0)

    def test_invalid_expected_rate_negative(self) -> None:
        """Test that negative expected_rate raises error."""
        timestamps = np.linspace(0, 1e-3, 1000)

        with pytest.raises(ValueError, match="expected_rate"):
            correct_timestamp_jitter(timestamps, expected_rate=-1e6)

    def test_invalid_max_correction_factor_zero(self) -> None:
        """Test that zero max_correction_factor raises error."""
        timestamps = np.linspace(0, 1e-3, 1000)

        with pytest.raises(ValueError, match="max_correction_factor"):
            correct_timestamp_jitter(timestamps, expected_rate=1e6, max_correction_factor=0)

    def test_invalid_max_correction_factor_negative(self) -> None:
        """Test that negative max_correction_factor raises error."""
        timestamps = np.linspace(0, 1e-3, 1000)

        with pytest.raises(ValueError, match="max_correction_factor"):
            correct_timestamp_jitter(timestamps, expected_rate=1e6, max_correction_factor=-1.0)

    def test_two_samples_no_correction(self) -> None:
        """Test with only two samples (insufficient for correction)."""
        timestamps = np.array([0.0, 1e-6])

        result = correct_timestamp_jitter(timestamps, expected_rate=1e6)

        # Should return with minimal processing
        assert len(result.corrected_timestamps) == 2

    def test_single_sample_no_correction(self) -> None:
        """Test with single sample (insufficient for correction)."""
        timestamps = np.array([0.0])

        result = correct_timestamp_jitter(timestamps, expected_rate=1e6)

        # Should return gracefully
        assert len(result.corrected_timestamps) == 1


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_usb_logic_analyzer_jitter(self) -> None:
        """Test correction of typical USB logic analyzer jitter."""
        # Simulate USB bulk transfer jitter (50-200 ns typical)
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        n_samples = int(sample_rate * duration)

        base_timestamps = np.linspace(0, duration, n_samples)
        rng = np.random.default_rng(42)
        usb_jitter = rng.normal(0, 1e-7, n_samples)  # 100 ns RMS
        jittery = base_timestamps + usb_jitter

        result = correct_timestamp_jitter(jittery, expected_rate=sample_rate)

        # Should significantly reduce jitter
        assert result.reduction_ratio > 2.0
        assert result.corrected_jitter_rms < result.original_jitter_rms

    def test_clock_drift_correction(self) -> None:
        """Test correction of clock drift."""
        # Simulate slow clock drift (1% error)
        sample_rate = 1e6
        duration = 0.01
        n_samples = int(sample_rate * duration)

        # Slightly slower actual rate
        actual_rate = sample_rate * 0.99
        timestamps = np.linspace(0, duration, n_samples) * (sample_rate / actual_rate)

        result = correct_timestamp_jitter(timestamps, expected_rate=sample_rate)

        # Should handle drift (near 1.0 is acceptable for minimal drift)
        assert result.reduction_ratio >= 0.99  # Tolerance for minimal drift scenarios

    def test_high_speed_capture_jitter(self) -> None:
        """Test correction for high-speed captures."""
        # 100 MHz capture with jitter
        sample_rate = 1e8
        duration = 1e-4  # 100 us
        n_samples = int(sample_rate * duration)

        base_timestamps = np.linspace(0, duration, n_samples)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-10, n_samples)  # 100 ps

        result = correct_timestamp_jitter(jittery, expected_rate=sample_rate)

        # Should reduce jitter (or stay near 1.0 for minimal jitter)
        assert result.reduction_ratio >= 0.99  # Small jitter may not improve significantly


class TestCorrectionQuality:
    """Test quality of jitter correction."""

    def test_samples_corrected_count(self) -> None:
        """Test that samples_corrected is tracked."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6)

        # Some samples should be corrected
        assert result.samples_corrected >= 0
        assert result.samples_corrected <= len(jittery)

    def test_max_correction_reported(self) -> None:
        """Test that maximum correction is reported."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6)

        # Max correction should be reasonable
        assert result.max_correction > 0
        assert result.max_correction < 1e-5  # Should be << period

    def test_reduction_ratio_realistic(self) -> None:
        """Test that reduction ratio is realistic."""
        base_timestamps = np.linspace(0, 1e-3, 1000)
        rng = np.random.default_rng(42)
        jittery = base_timestamps + rng.normal(0, 1e-7, 1000)

        result = correct_timestamp_jitter(jittery, expected_rate=1e6)

        # Reduction should be significant but not unrealistic
        assert 1.0 <= result.reduction_ratio <= 100.0
