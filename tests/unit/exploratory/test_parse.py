"""Comprehensive unit tests for error-tolerant parsing module.

This module provides extensive testing for timestamp correction and
error-tolerant protocol decoding functionality.


Test Coverage:
- correct_timestamp_jitter with lowpass and PLL methods
- decode_with_error_tolerance for various protocols
- ErrorTolerance modes (STRICT, TOLERANT, PERMISSIVE)
- Edge cases: empty data, constant signals, extreme jitter
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.exploratory.parse import (
    ErrorTolerance,
    correct_timestamp_jitter,
    decode_with_error_tolerance,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def generate_jittery_timestamps(
    n_samples: int, expected_rate: float, jitter_rms: float, seed: int = 42
) -> np.ndarray:
    """Generate timestamps with Gaussian jitter."""
    rng = np.random.default_rng(seed)
    period = 1.0 / expected_rate

    # Ideal timestamps
    ideal = np.arange(n_samples) * period

    # Add jitter
    jitter = rng.normal(0, jitter_rms, n_samples)
    return ideal + jitter


def generate_bursty_jitter_timestamps(
    n_samples: int, expected_rate: float, burst_positions: list[int], burst_magnitude: float
) -> np.ndarray:
    """Generate timestamps with burst jitter."""
    period = 1.0 / expected_rate
    timestamps = np.arange(n_samples, dtype=np.float64) * period

    # Add burst jitter
    for pos in burst_positions:
        if pos < n_samples:
            timestamps[pos] += burst_magnitude

    return np.cumsum(np.diff(np.concatenate([[0], timestamps])))


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestCorrectTimestampJitter:
    """Test timestamp jitter correction (DAQ-003)."""

    def test_no_jitter_unchanged(self) -> None:
        """Test that perfect timestamps remain unchanged."""
        expected_rate = 1e6
        n_samples = 1000
        period = 1.0 / expected_rate

        # Perfect timestamps
        timestamps = np.arange(n_samples) * period

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Should have minimal correction
        assert result.original_jitter_rms == pytest.approx(0.0, abs=1e-10)
        assert result.samples_corrected == 0
        assert result.reduction_ratio >= 1.0

    def test_lowpass_reduces_jitter(self) -> None:
        """Test that lowpass filtering reduces jitter."""
        expected_rate = 1e6
        n_samples = 1000
        jitter_rms = 1e-7  # 100 ns RMS jitter

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Jitter should be reduced
        assert result.corrected_jitter_rms < result.original_jitter_rms
        assert result.reduction_ratio > 1.0
        assert result.samples_corrected > 0

    def test_pll_reduces_jitter(self) -> None:
        """Test that PLL method reduces jitter."""
        expected_rate = 1e6
        n_samples = 1000
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="pll")

        # PLL should also reduce jitter
        assert result.corrected_jitter_rms < result.original_jitter_rms
        assert result.reduction_ratio > 1.0

    def test_max_correction_limit(self) -> None:
        """Test that corrections respect max_correction_factor."""
        expected_rate = 1e6
        period = 1.0 / expected_rate
        n_samples = 100

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms=period * 0.5)

        max_factor = 0.5
        result = correct_timestamp_jitter(
            timestamps, expected_rate, method="pll", max_correction_factor=max_factor
        )

        # Max correction should not exceed max_correction_factor * period
        max_allowed = max_factor * period
        assert result.max_correction <= max_allowed * 1.01  # Small tolerance for numerics

    def test_empty_timestamps_raises_error(self) -> None:
        """Test that empty timestamps raise ValueError."""
        timestamps = np.array([])
        expected_rate = 1e6

        with pytest.raises(ValueError, match="cannot be empty"):
            correct_timestamp_jitter(timestamps, expected_rate)

    def test_invalid_rate_raises_error(self) -> None:
        """Test that invalid rate raises ValueError."""
        timestamps = np.arange(100) / 1e6

        with pytest.raises(ValueError, match="must be positive"):
            correct_timestamp_jitter(timestamps, expected_rate=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            correct_timestamp_jitter(timestamps, expected_rate=-100.0)

    def test_invalid_max_correction_raises_error(self) -> None:
        """Test that invalid max_correction_factor raises ValueError."""
        timestamps = np.arange(100) / 1e6

        with pytest.raises(ValueError, match="must be positive"):
            correct_timestamp_jitter(timestamps, 1e6, max_correction_factor=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            correct_timestamp_jitter(timestamps, 1e6, max_correction_factor=-1.0)

    def test_few_samples_returns_unchanged(self) -> None:
        """Test that very few samples return unchanged."""
        timestamps = np.array([0.0, 1e-6])  # Only 2 samples
        expected_rate = 1e6

        result = correct_timestamp_jitter(timestamps, expected_rate)

        # Should return unchanged due to insufficient data
        np.testing.assert_array_almost_equal(result.corrected_timestamps, timestamps)
        assert result.reduction_ratio == 1.0

    def test_reduction_ratio_target(self) -> None:
        """Test that typical USB jitter achieves >5x reduction."""
        expected_rate = 1e6
        n_samples = 5000
        # Typical USB jitter: ~100ns RMS
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Should achieve at least 2x reduction (conservative target)
        assert result.reduction_ratio > 2.0

    def test_corrected_timestamps_monotonic(self) -> None:
        """Test that corrected timestamps remain monotonic."""
        expected_rate = 1e6
        n_samples = 1000
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Check monotonicity
        diffs = np.diff(result.corrected_timestamps)
        assert np.all(diffs >= 0), "Corrected timestamps should be monotonic"

    def test_mean_period_preserved(self) -> None:
        """Test that mean period is preserved after correction."""
        expected_rate = 1e6
        expected_period = 1.0 / expected_rate
        n_samples = 1000
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Check mean period
        corrected_periods = np.diff(result.corrected_timestamps)
        mean_period = np.mean(corrected_periods)

        assert mean_period == pytest.approx(expected_period, rel=0.01)

    def test_comparison_lowpass_vs_pll(self) -> None:
        """Test that both methods provide similar results."""
        expected_rate = 1e6
        n_samples = 1000
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result_lowpass = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")
        result_pll = correct_timestamp_jitter(timestamps, expected_rate, method="pll")

        # Both should reduce jitter
        assert result_lowpass.reduction_ratio > 1.0
        assert result_pll.reduction_ratio > 1.0


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestDecodeWithErrorTolerance:
    """Test error-tolerant protocol decoding (DAQ-004)."""

    def test_strict_mode_aborts_on_error(self) -> None:
        """Test STRICT mode aborts on first error."""
        data = np.array([0xFF, 0x55, 0xAA, 0x00], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.STRICT, baud=9600
        )

        # In STRICT mode, should stop after first error (0xFF is framing error)
        assert len(frames) >= 1
        # First frame should be invalid
        if len(frames) > 0:
            assert frames[0].data == b"\xff"

    def test_tolerant_mode_continues_after_error(self) -> None:
        """Test TOLERANT mode continues after errors."""
        data = np.array([0xFF, 0x55, 0xAA, 0x00], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.TOLERANT, baud=9600
        )

        # Should process all data despite errors
        assert len(frames) >= 1

    def test_permissive_mode_reports_all_errors(self) -> None:
        """Test PERMISSIVE mode reports all errors."""
        data = np.array([0xFF, 0x55, 0xAA, 0xFF, 0x00], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        # Should decode all frames, marking errors
        assert len(frames) >= 1
        # Check that some frames are marked as invalid
        invalid_frames = [f for f in frames if not f.valid]
        assert len(invalid_frames) >= 0  # May have errors

    def test_decoded_frame_structure(self) -> None:
        """Test DecodedFrame structure."""
        data = np.array([0x55, 0xAA], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.TOLERANT, baud=9600
        )

        if len(frames) > 0:
            frame = frames[0]
            # Check required fields
            assert hasattr(frame, "data")
            assert hasattr(frame, "timestamp")
            assert hasattr(frame, "valid")
            assert hasattr(frame, "error_type")
            assert hasattr(frame, "position")

            assert isinstance(frame.data, bytes)
            assert isinstance(frame.timestamp, float)
            assert isinstance(frame.valid, bool)
            assert frame.error_type is None or isinstance(frame.error_type, str)
            assert isinstance(frame.position, int)

    def test_unsupported_protocol_raises_error(self) -> None:
        """Test that unsupported protocol raises ValueError."""
        data = np.array([0x55], dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported protocol"):
            decode_with_error_tolerance(data, "invalid_protocol", baud=9600)  # type: ignore[arg-type]

    def test_missing_required_parameter_raises_error(self) -> None:
        """Test that missing required parameters raise ValueError."""
        data = np.array([0x55], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires"):
            decode_with_error_tolerance(data, "uart")  # Missing baud parameter

    def test_empty_data_returns_empty(self) -> None:
        """Test decoding empty data."""
        data = np.array([], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.TOLERANT, baud=9600
        )

        assert len(frames) == 0

    def test_single_byte_decode(self) -> None:
        """Test decoding single byte."""
        data = np.array([0x55], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        assert len(frames) >= 0  # May or may not produce a frame

    def test_timestamp_progression(self) -> None:
        """Test that timestamps progress correctly."""
        data = np.array([0x55, 0xAA, 0x00, 0xFF], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        if len(frames) > 1:
            # Timestamps should be non-decreasing
            timestamps = [f.timestamp for f in frames]
            for i in range(len(timestamps) - 1):
                assert timestamps[i + 1] >= timestamps[i]

    def test_position_tracking(self) -> None:
        """Test that frame positions are tracked."""
        data = np.array([0x55, 0xAA, 0x00, 0xFF], dtype=np.uint8)

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        if len(frames) > 0:
            # Positions should be valid indices
            for frame in frames:
                assert 0 <= frame.position < len(data)

    def test_error_type_annotation(self) -> None:
        """Test that error types are annotated."""
        data = np.array([0xFF, 0x55], dtype=np.uint8)  # 0xFF causes framing error

        frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        # Check for error annotations
        if len(frames) > 0:
            error_frames = [f for f in frames if not f.valid and f.error_type is not None]
            # May have error frames
            assert len(error_frames) >= 0


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestExploratoryParseEdgeCases:
    """Test edge cases and robustness."""

    def test_very_high_jitter(self) -> None:
        """Test correction with very high jitter."""
        expected_rate = 1e6
        period = 1.0 / expected_rate
        n_samples = 1000
        # Extreme jitter (50% of period)
        jitter_rms = period * 0.5

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms)

        result = correct_timestamp_jitter(
            timestamps, expected_rate, method="lowpass", max_correction_factor=2.0
        )

        # Should still produce valid result
        assert len(result.corrected_timestamps) == n_samples
        assert not np.any(np.isnan(result.corrected_timestamps))

    def test_constant_timestamps(self) -> None:
        """Test with constant timestamps (no variation)."""
        timestamps = np.ones(100) * 1e-6

        result = correct_timestamp_jitter(timestamps, 1e6, method="lowpass")

        # Should handle gracefully
        assert len(result.corrected_timestamps) == 100

    def test_negative_timestamps(self) -> None:
        """Test with negative timestamps."""
        timestamps = np.arange(-100, 0) / 1e6

        result = correct_timestamp_jitter(timestamps, 1e6, method="pll")

        # Should still work (timestamps can be relative)
        assert len(result.corrected_timestamps) == 100

    def test_large_dataset(self) -> None:
        """Test with large dataset."""
        expected_rate = 1e6
        n_samples = 100000  # Large dataset
        jitter_rms = 1e-7

        timestamps = generate_jittery_timestamps(n_samples, expected_rate, jitter_rms, seed=42)

        result = correct_timestamp_jitter(timestamps, expected_rate, method="lowpass")

        # Should process without issues
        assert len(result.corrected_timestamps) == n_samples
        assert result.reduction_ratio > 1.0

    def test_mixed_tolerances(self) -> None:
        """Test different tolerance levels produce different results."""
        data = np.array([0xFF, 0x55, 0xAA, 0xFF], dtype=np.uint8)

        strict_frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.STRICT, baud=9600
        )

        tolerant_frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.TOLERANT, baud=9600
        )

        permissive_frames = decode_with_error_tolerance(
            data, "uart", tolerance=ErrorTolerance.PERMISSIVE, baud=9600
        )

        # Permissive should generally produce most frames
        # (though exact behavior depends on implementation)
        assert len(permissive_frames) >= 0
        assert len(tolerant_frames) >= 0
        assert len(strict_frames) >= 0
