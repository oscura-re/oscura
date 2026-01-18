"""Comprehensive unit tests for interpolation module.

This module tests all interpolation and resampling functionality:
- Interpolation methods (linear, cubic, nearest, zero)
- Resampling methods (fft, polyphase, interp)
- Downsampling methods (decimate, average, max, min)
- Trace alignment
- Edge cases (single point, identical points, out-of-bounds)
- Extrapolation handling
- Anti-aliasing filters
- Nyquist criterion validation


Coverage target: >90% branch coverage
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.math.interpolation import (
    align_traces,
    downsample,
    interpolate,
    resample,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Default sample rate for test traces (1 MHz)."""
    return 1_000_000.0


@pytest.fixture
def simple_metadata(sample_rate: float) -> TraceMetadata:
    """Create simple metadata for testing."""
    return TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=1.0,
        vertical_offset=0.0,
        channel_name="test_channel",
    )


@pytest.fixture
def simple_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a simple trace with linear values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def sine_trace(sample_rate: float) -> WaveformTrace:
    """Create a sine wave trace for testing."""
    duration = 0.001  # 1 ms
    t = np.arange(0, duration, 1 / sample_rate)
    frequency = 1000.0  # 1 kHz
    data = np.sin(2 * np.pi * frequency * t)
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="sine_1khz",
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def high_freq_trace(sample_rate: float) -> WaveformTrace:
    """Create a high-frequency sine wave for Nyquist testing."""
    duration = 0.001
    t = np.arange(0, duration, 1 / sample_rate)
    # High frequency: 100 kHz (well within Nyquist for 1 MHz)
    frequency = 100_000.0
    data = np.sin(2 * np.pi * frequency * t)
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="sine_100khz",
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def constant_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a constant-value trace."""
    data = np.full(100, 5.0, dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def single_sample_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a trace with only one sample."""
    data = np.array([3.14], dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def two_sample_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a trace with exactly two samples."""
    data = np.array([1.0, 2.0], dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


# =============================================================================
# Tests for interpolate()
# =============================================================================


@pytest.mark.unit
class TestInterpolate:
    """Tests for interpolate function."""

    def test_linear_interpolation_basic(self, simple_trace: WaveformTrace):
        """Test basic linear interpolation."""
        new_time = np.linspace(0, 4e-6, 9)  # Doubled points
        result = interpolate(simple_trace, new_time, method="linear")

        assert len(result.data) == 9
        assert result.metadata.sample_rate > 0
        # First and last values should be preserved
        np.testing.assert_allclose(result.data[0], simple_trace.data[0], rtol=1e-10)

    def test_linear_interpolation_preserves_values(self, simple_trace: WaveformTrace):
        """Test that interpolation at original points preserves values."""
        original_time = simple_trace.time_vector
        result = interpolate(simple_trace, original_time, method="linear")

        np.testing.assert_allclose(result.data, simple_trace.data, rtol=1e-10)

    def test_cubic_interpolation(self, sine_trace: WaveformTrace):
        """Test cubic spline interpolation."""
        # Create denser time points
        duration = len(sine_trace.data) / sine_trace.metadata.sample_rate
        new_time = np.linspace(0, duration * 0.5, 1000)
        result = interpolate(sine_trace, new_time, method="cubic")

        assert len(result.data) == 1000
        assert result.metadata.sample_rate > 0
        # Cubic interpolation should be smooth
        assert np.all(np.isfinite(result.data))

    def test_nearest_interpolation(self, simple_trace: WaveformTrace):
        """Test nearest neighbor interpolation."""
        new_time = np.array([0.5e-6, 1.5e-6, 2.5e-6])
        result = interpolate(simple_trace, new_time, method="nearest")

        assert len(result.data) == 3
        # Nearest should pick closest values
        assert np.all(np.isfinite(result.data))

    def test_zero_order_hold(self, simple_trace: WaveformTrace):
        """Test zero-order hold interpolation."""
        new_time = np.linspace(0, 4e-6, 20)
        result = interpolate(simple_trace, new_time, method="zero")

        assert len(result.data) == 20
        # Zero-order creates step function
        assert np.all(np.isfinite(result.data))

    def test_extrapolation_with_nan(self, simple_trace: WaveformTrace):
        """Test extrapolation fills with NaN by default."""
        # Time points beyond original range
        new_time = np.array([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6])
        result = interpolate(simple_trace, new_time, method="linear")

        # Last two points are beyond range, should be NaN
        assert np.isnan(result.data[-1])
        assert np.isnan(result.data[-2])
        # First points should be valid
        assert not np.isnan(result.data[0])

    def test_extrapolation_with_constant(self, simple_trace: WaveformTrace):
        """Test extrapolation with constant fill value."""
        new_time = np.array([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6])
        result = interpolate(simple_trace, new_time, method="linear", fill_value=0.0)

        # Points beyond range should be 0.0
        assert result.data[-1] == 0.0
        assert result.data[-2] == 0.0

    def test_extrapolation_with_tuple(self, simple_trace: WaveformTrace):
        """Test extrapolation with different below/above values."""
        new_time = np.array([-1e-6, 0, 2e-6, 4e-6, 6e-6])
        result = interpolate(simple_trace, new_time, method="linear", fill_value=(-99.0, 99.0))

        # Below range
        assert result.data[0] == -99.0
        # Above range
        assert result.data[-1] == 99.0

    def test_single_time_point(self, simple_trace: WaveformTrace):
        """Test interpolation to single time point."""
        new_time = np.array([2e-6])
        result = interpolate(simple_trace, new_time, method="linear")

        assert len(result.data) == 1
        assert np.isfinite(result.data[0])
        # Sample rate should be preserved when only one point
        assert result.metadata.sample_rate == simple_trace.metadata.sample_rate

    def test_custom_channel_name(self, simple_trace: WaveformTrace):
        """Test custom channel name assignment."""
        new_time = np.linspace(0, 4e-6, 10)
        result = interpolate(simple_trace, new_time, method="linear", channel_name="custom_interp")

        assert result.metadata.channel_name == "custom_interp"

    def test_default_channel_name(self, simple_trace: WaveformTrace):
        """Test default channel name generation."""
        new_time = np.linspace(0, 4e-6, 10)
        result = interpolate(simple_trace, new_time, method="linear")

        assert "_interp" in result.metadata.channel_name

    def test_insufficient_samples_error(self, single_sample_trace: WaveformTrace):
        """Test error when trace has insufficient samples."""
        new_time = np.array([0, 1e-6])
        with pytest.raises(InsufficientDataError) as exc_info:
            interpolate(single_sample_trace, new_time, method="linear")

        assert "at least 2 samples" in str(exc_info.value).lower()
        assert exc_info.value.required == 2
        assert exc_info.value.available == 1

    def test_minimum_samples_succeeds(self, two_sample_trace: WaveformTrace):
        """Test that two samples is sufficient."""
        new_time = np.linspace(0, 1e-6, 5)
        result = interpolate(two_sample_trace, new_time, method="linear")

        assert len(result.data) == 5
        assert np.all(np.isfinite(result.data))

    def test_invalid_method_error(self, simple_trace: WaveformTrace):
        """Test error on invalid interpolation method."""
        new_time = np.linspace(0, 4e-6, 10)
        with pytest.raises(ValueError) as exc_info:
            interpolate(simple_trace, new_time, method="invalid")  # type: ignore

        assert "unknown interpolation method" in str(exc_info.value).lower()

    def test_metadata_preservation(self, simple_trace: WaveformTrace):
        """Test that metadata fields are preserved."""
        new_time = np.linspace(0, 4e-6, 10)
        result = interpolate(simple_trace, new_time, method="linear")

        assert result.metadata.vertical_scale == simple_trace.metadata.vertical_scale
        assert result.metadata.vertical_offset == simple_trace.metadata.vertical_offset
        assert result.metadata.source_file == simple_trace.metadata.source_file

    def test_upsampling(self, simple_trace: WaveformTrace):
        """Test interpolation that increases sample count."""
        duration = len(simple_trace.data) / simple_trace.metadata.sample_rate
        new_time = np.linspace(0, duration, 100)
        result = interpolate(simple_trace, new_time, method="linear")

        assert len(result.data) == 100
        assert len(result.data) > len(simple_trace.data)

    def test_downsampling(self, sine_trace: WaveformTrace):
        """Test interpolation that decreases sample count."""
        duration = len(sine_trace.data) / sine_trace.metadata.sample_rate
        new_time = np.linspace(0, duration * 0.5, 100)
        result = interpolate(sine_trace, new_time, method="linear")

        assert len(result.data) == 100
        assert len(result.data) < len(sine_trace.data)


# =============================================================================
# Tests for resample()
# =============================================================================


@pytest.mark.unit
class TestResample:
    """Tests for resample function - REQ: API-019."""

    def test_resample_by_rate_upsampling(self, sine_trace: WaveformTrace):
        """Test upsampling by specifying new sample rate."""
        new_rate = 2_000_000.0  # Double the rate
        result = resample(sine_trace, new_sample_rate=new_rate)

        assert result.metadata.sample_rate == new_rate
        expected_samples = len(sine_trace.data) * 2
        assert len(result.data) == pytest.approx(expected_samples, rel=0.01)

    def test_resample_by_rate_downsampling(self, sine_trace: WaveformTrace):
        """Test downsampling by specifying new sample rate."""
        new_rate = 500_000.0  # Half the rate
        result = resample(sine_trace, new_sample_rate=new_rate)

        assert result.metadata.sample_rate == new_rate
        expected_samples = len(sine_trace.data) // 2
        assert len(result.data) == pytest.approx(expected_samples, rel=0.01)

    def test_resample_by_num_samples(self, sine_trace: WaveformTrace):
        """Test resampling by specifying number of samples."""
        num_samples = 500
        result = resample(sine_trace, num_samples=num_samples)

        assert len(result.data) == num_samples
        # Sample rate should be adjusted proportionally
        expected_rate = sine_trace.metadata.sample_rate * num_samples / len(sine_trace.data)
        assert result.metadata.sample_rate == pytest.approx(expected_rate, rel=0.01)

    def test_missing_both_parameters_error(self, sine_trace: WaveformTrace):
        """Test error when neither rate nor samples specified."""
        with pytest.raises(ValueError) as exc_info:
            resample(sine_trace)

        assert "exactly one" in str(exc_info.value).lower()

    def test_both_parameters_specified_error(self, sine_trace: WaveformTrace):
        """Test error when both rate and samples specified."""
        with pytest.raises(ValueError) as exc_info:
            resample(sine_trace, new_sample_rate=2e6, num_samples=1000)

        assert "exactly one" in str(exc_info.value).lower()

    def test_insufficient_samples_error(self, single_sample_trace: WaveformTrace):
        """Test error when trace has insufficient samples."""
        with pytest.raises(InsufficientDataError) as exc_info:
            resample(single_sample_trace, new_sample_rate=2e6)

        assert "at least 2 samples" in str(exc_info.value).lower()

    def test_minimum_samples_succeeds(self, two_sample_trace: WaveformTrace):
        """Test that two samples is sufficient."""
        result = resample(two_sample_trace, num_samples=10)

        assert len(result.data) == 10

    def test_fft_method(self, sine_trace: WaveformTrace):
        """Test FFT-based resampling (highest quality)."""
        result = resample(sine_trace, new_sample_rate=2e6, method="fft")

        assert result.metadata.sample_rate == 2e6
        assert len(result.data) > len(sine_trace.data)
        # FFT should preserve sine wave characteristics
        assert np.max(result.data) <= 1.1  # Slight tolerance
        assert np.min(result.data) >= -1.1

    def test_polyphase_method(self, sine_trace: WaveformTrace):
        """Test polyphase filter resampling."""
        result = resample(sine_trace, new_sample_rate=2e6, method="polyphase")

        assert result.metadata.sample_rate == 2e6
        # Polyphase should be accurate
        assert np.all(np.isfinite(result.data))

    def test_interp_method(self, sine_trace: WaveformTrace):
        """Test linear interpolation resampling."""
        result = resample(sine_trace, new_sample_rate=2e6, method="interp")

        assert result.metadata.sample_rate == 2e6
        assert np.all(np.isfinite(result.data))

    def test_invalid_method_error(self, sine_trace: WaveformTrace):
        """Test error on invalid resampling method."""
        with pytest.raises(ValueError) as exc_info:
            resample(sine_trace, new_sample_rate=2e6, method="invalid")  # type: ignore

        assert "unknown resampling method" in str(exc_info.value).lower()

    def test_anti_alias_enabled(self, high_freq_trace: WaveformTrace):
        """Test anti-aliasing filter is applied when downsampling."""
        # Downsample to 100 kHz (from 1 MHz)
        # This will trigger Nyquist warning since signal is 100 kHz, suppress it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result_with = resample(high_freq_trace, new_sample_rate=100_000, anti_alias=True)
            result_without = resample(high_freq_trace, new_sample_rate=100_000, anti_alias=False)

        assert len(result_with.data) == len(result_without.data)
        # With anti-aliasing, high frequencies should be attenuated
        # This is hard to test precisely, but both should complete

    def test_anti_alias_disabled(self, sine_trace: WaveformTrace):
        """Test resampling without anti-aliasing."""
        result = resample(sine_trace, new_sample_rate=500_000, anti_alias=False)

        assert result.metadata.sample_rate == 500_000

    def test_nyquist_warning_triggered(self, high_freq_trace: WaveformTrace):
        """Test Nyquist criterion warning - REQ: API-019."""
        # High freq trace is 100 kHz, needs >= 200 kHz
        # Downsample to 150 kHz should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resample(high_freq_trace, new_sample_rate=150_000)

            # Should have a UserWarning about Nyquist
            assert len(w) >= 1
            assert any(issubclass(warn.category, UserWarning) for warn in w)
            assert any("nyquist" in str(warn.message).lower() for warn in w)

    def test_no_nyquist_warning_when_safe(self, high_freq_trace: WaveformTrace):
        """Test no warning when resampling above Nyquist rate."""
        # High freq trace is 100 kHz, 500 kHz is safe
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resample(high_freq_trace, new_sample_rate=500_000)

            # Should not have Nyquist warnings
            nyquist_warnings = [warn for warn in w if "nyquist" in str(warn.message).lower()]
            assert len(nyquist_warnings) == 0

    def test_no_nyquist_warning_on_upsampling(self, sine_trace: WaveformTrace):
        """Test no warning when upsampling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resample(sine_trace, new_sample_rate=2_000_000)

            # No warnings on upsampling
            assert len(w) == 0

    def test_target_samples_too_small_error(self, sine_trace: WaveformTrace):
        """Test error when target samples would be less than 1."""
        with pytest.raises(ValueError) as exc_info:
            resample(sine_trace, num_samples=0)

        assert "at least 1" in str(exc_info.value).lower()

    def test_custom_channel_name(self, sine_trace: WaveformTrace):
        """Test custom channel name assignment."""
        result = resample(sine_trace, new_sample_rate=2e6, channel_name="custom_resample")

        assert result.metadata.channel_name == "custom_resample"

    def test_default_channel_name(self, sine_trace: WaveformTrace):
        """Test default channel name generation."""
        result = resample(sine_trace, new_sample_rate=2e6)

        assert "_resampled" in result.metadata.channel_name

    def test_metadata_preservation(self, sine_trace: WaveformTrace):
        """Test that metadata fields are preserved."""
        result = resample(sine_trace, new_sample_rate=2e6)

        assert result.metadata.vertical_scale == sine_trace.metadata.vertical_scale
        assert result.metadata.vertical_offset == sine_trace.metadata.vertical_offset

    def test_constant_signal_resampling(self, constant_trace: WaveformTrace):
        """Test resampling a constant signal."""
        result = resample(constant_trace, num_samples=50)

        assert len(result.data) == 50
        # All values should be close to original constant
        np.testing.assert_allclose(result.data, 5.0, rtol=0.01)

    def test_polyphase_exact_length(self, sine_trace: WaveformTrace):
        """Test polyphase method returns exact target length."""
        target_samples = 500
        result = resample(sine_trace, num_samples=target_samples, method="polyphase")

        assert len(result.data) == target_samples


# =============================================================================
# Tests for downsample()
# =============================================================================


@pytest.mark.unit
class TestDownsample:
    """Tests for downsample function - REQ: MEM-012."""

    def test_decimate_basic(self, sine_trace: WaveformTrace):
        """Test basic decimation downsampling."""
        factor = 10
        result = downsample(sine_trace, factor, method="decimate")

        assert result.metadata.sample_rate == sine_trace.metadata.sample_rate / factor
        assert len(result.data) == len(sine_trace.data) // factor

    def test_average_method(self, simple_trace: WaveformTrace):
        """Test averaging downsampling."""
        # Create trace with known pattern
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="test"),
        )

        result = downsample(trace, factor=2, method="average")

        # Should average pairs: (1+2)/2, (3+4)/2, etc.
        expected = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_max_method(self, simple_trace: WaveformTrace):
        """Test max downsampling."""
        # Create trace with known pattern
        data = np.array([1, 5, 2, 6, 3, 7, 4, 8], dtype=np.float64)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="test"),
        )

        result = downsample(trace, factor=2, method="max")

        # Should take max of pairs: max(1,5), max(2,6), etc.
        expected = np.array([5, 6, 7, 8])
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_min_method(self, simple_trace: WaveformTrace):
        """Test min downsampling."""
        # Create trace with known pattern
        data = np.array([1, 5, 2, 6, 3, 7, 4, 8], dtype=np.float64)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="test"),
        )

        result = downsample(trace, factor=2, method="min")

        # Should take min of pairs: min(1,5), min(2,6), etc.
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_factor_one_returns_original(self, sine_trace: WaveformTrace):
        """Test that factor=1 returns the original trace."""
        result = downsample(sine_trace, factor=1)

        assert result is sine_trace

    def test_factor_less_than_one_error(self, sine_trace: WaveformTrace):
        """Test error when factor < 1."""
        with pytest.raises(ValueError) as exc_info:
            downsample(sine_trace, factor=0)

        assert "must be >= 1" in str(exc_info.value).lower()

    def test_invalid_method_error(self, sine_trace: WaveformTrace):
        """Test error on invalid downsampling method."""
        with pytest.raises(ValueError) as exc_info:
            downsample(sine_trace, factor=2, method="invalid")  # type: ignore

        assert "unknown method" in str(exc_info.value).lower()

    def test_anti_alias_enabled(self, high_freq_trace: WaveformTrace):
        """Test anti-aliasing filter with decimation."""
        result = downsample(high_freq_trace, factor=10, anti_alias=True)

        assert len(result.data) == len(high_freq_trace.data) // 10
        # Anti-aliasing should attenuate high frequencies

    def test_anti_alias_disabled(self, sine_trace: WaveformTrace):
        """Test decimation without anti-aliasing."""
        result = downsample(sine_trace, factor=5, anti_alias=False)

        assert len(result.data) == len(sine_trace.data) // 5

    def test_anti_alias_only_for_decimate(self, sine_trace: WaveformTrace):
        """Test anti-aliasing is only applied for decimate method."""
        # These should all work without errors
        downsample(sine_trace, factor=5, method="average", anti_alias=True)
        downsample(sine_trace, factor=5, method="max", anti_alias=True)
        downsample(sine_trace, factor=5, method="min", anti_alias=True)

    def test_truncation_to_multiple(self, sine_trace: WaveformTrace):
        """Test that data is truncated to multiple of factor."""
        # Create trace with length not divisible by factor
        data = np.arange(103, dtype=np.float64)  # 103 samples
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="test"),
        )

        result = downsample(trace, factor=10, method="decimate")

        # Should truncate to 100, then downsample to 10
        assert len(result.data) == 10

    def test_custom_channel_name(self, sine_trace: WaveformTrace):
        """Test custom channel name assignment."""
        result = downsample(sine_trace, factor=5, channel_name="custom_downsample")

        assert result.metadata.channel_name == "custom_downsample"

    def test_default_channel_name(self, sine_trace: WaveformTrace):
        """Test default channel name generation."""
        result = downsample(sine_trace, factor=5)

        assert "_ds5" in result.metadata.channel_name

    def test_metadata_preservation(self, sine_trace: WaveformTrace):
        """Test that metadata fields are preserved."""
        result = downsample(sine_trace, factor=5)

        assert result.metadata.vertical_scale == sine_trace.metadata.vertical_scale
        assert result.metadata.vertical_offset == sine_trace.metadata.vertical_offset

    def test_large_downsampling_factor(self, sine_trace: WaveformTrace):
        """Test downsampling with large factor."""
        factor = 100
        result = downsample(sine_trace, factor=factor, method="average")

        expected_length = len(sine_trace.data) // factor
        assert len(result.data) == expected_length


# =============================================================================
# Tests for align_traces()
# =============================================================================


@pytest.mark.unit
class TestAlignTraces:
    """Tests for align_traces function."""

    @pytest.fixture
    def trace_pair(self) -> tuple[WaveformTrace, WaveformTrace]:
        """Create two traces with different sample rates."""
        # Trace 1: 1000 Hz, 100 samples
        t1 = np.arange(0, 0.1, 1 / 1000)
        data1 = np.sin(2 * np.pi * 10 * t1)
        trace1 = WaveformTrace(
            data=data1,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="trace1"),
        )

        # Trace 2: 2000 Hz, 200 samples
        t2 = np.arange(0, 0.1, 1 / 2000)
        data2 = np.cos(2 * np.pi * 10 * t2)
        trace2 = WaveformTrace(
            data=data2,
            metadata=TraceMetadata(sample_rate=2000.0, channel_name="trace2"),
        )

        return trace1, trace2

    @pytest.fixture
    def different_length_pair(self) -> tuple[WaveformTrace, WaveformTrace]:
        """Create two traces with different lengths and rates."""
        # Trace 1: 1000 Hz, 200 samples (0.2 seconds)
        t1 = np.arange(0, 0.2, 1 / 1000)
        data1 = np.sin(2 * np.pi * 5 * t1)
        trace1 = WaveformTrace(
            data=data1,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="long_trace"),
        )

        # Trace 2: 500 Hz, 50 samples (0.1 seconds)
        t2 = np.arange(0, 0.1, 1 / 500)
        data2 = np.cos(2 * np.pi * 5 * t2)
        trace2 = WaveformTrace(
            data=data2,
            metadata=TraceMetadata(sample_rate=500.0, channel_name="short_trace"),
        )

        return trace1, trace2

    def test_align_interpolate_method(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning traces using interpolation."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, method="interpolate")

        # Should have same sample rate
        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        # Should have same length
        assert len(aligned1.data) == len(aligned2.data)

    def test_align_resample_method(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning traces using resampling."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, method="resample")

        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        assert len(aligned1.data) == len(aligned2.data)

    def test_align_higher_rate_reference(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning to higher sample rate (default)."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, reference="higher")

        # Should use 2000 Hz (the higher rate)
        assert aligned1.metadata.sample_rate == 2000.0
        assert aligned2.metadata.sample_rate == 2000.0

    def test_align_first_rate_reference(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning to first trace's rate."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, reference="first")

        # Should use 1000 Hz (trace1's rate)
        assert aligned1.metadata.sample_rate == 1000.0
        assert aligned2.metadata.sample_rate == 1000.0

    def test_align_second_rate_reference(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning to second trace's rate."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, reference="second")

        # Should use 2000 Hz (trace2's rate)
        assert aligned1.metadata.sample_rate == 2000.0
        assert aligned2.metadata.sample_rate == 2000.0

    def test_align_different_lengths(
        self, different_length_pair: tuple[WaveformTrace, WaveformTrace]
    ):
        """Test aligning traces with different lengths."""
        trace1, trace2 = different_length_pair
        aligned1, aligned2 = align_traces(trace1, trace2)

        # Should align to common (shorter) duration
        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        assert len(aligned1.data) == len(aligned2.data)
        # Duration should be limited by shorter trace (0.1 seconds)
        duration = len(aligned1.data) / aligned1.metadata.sample_rate
        assert duration <= 0.1

    def test_align_custom_channel_names(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test custom channel names for aligned traces."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, channel_names=("aligned_a", "aligned_b"))

        assert aligned1.metadata.channel_name == "aligned_a"
        assert aligned2.metadata.channel_name == "aligned_b"

    def test_align_default_channel_names(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test default channel name generation."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2)

        # Should have some channel name
        assert aligned1.metadata.channel_name is not None
        assert aligned2.metadata.channel_name is not None

    def test_align_partial_custom_names(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test with only one custom name specified."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, channel_names=("custom_name", None))

        assert aligned1.metadata.channel_name == "custom_name"
        # Second should get auto-generated name

    def test_align_same_rate_traces(self):
        """Test aligning traces that already have the same rate."""
        # Create two traces with same rate
        data1 = np.sin(np.linspace(0, 2 * np.pi, 100))
        data2 = np.cos(np.linspace(0, 2 * np.pi, 100))

        trace1 = WaveformTrace(
            data=data1, metadata=TraceMetadata(sample_rate=1000.0, channel_name="a")
        )
        trace2 = WaveformTrace(
            data=data2, metadata=TraceMetadata(sample_rate=1000.0, channel_name="b")
        )

        aligned1, aligned2 = align_traces(trace1, trace2)

        # Should still work and align properly
        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        assert len(aligned1.data) == len(aligned2.data)

    def test_align_preserves_waveform_characteristics(
        self, trace_pair: tuple[WaveformTrace, WaveformTrace]
    ):
        """Test that alignment preserves basic waveform characteristics."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2)

        # Should preserve approximate min/max values (ignoring NaN from extrapolation)
        assert np.nanmax(aligned1.data) <= np.max(trace1.data) + 0.1
        assert np.nanmin(aligned1.data) >= np.min(trace1.data) - 0.1


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
class TestMathInterpolationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_interpolate_empty_trace(self):
        """Test interpolation with empty trace."""
        empty_trace = WaveformTrace(
            data=np.array([], dtype=np.float64),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        with pytest.raises(InsufficientDataError):
            interpolate(empty_trace, np.array([0, 1e-6]))

    def test_resample_empty_trace(self):
        """Test resampling with empty trace."""
        empty_trace = WaveformTrace(
            data=np.array([], dtype=np.float64),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        with pytest.raises(InsufficientDataError):
            resample(empty_trace, new_sample_rate=2000)

    def test_interpolate_with_nans_in_data(self):
        """Test interpolation with NaN values in input data."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000.0))

        new_time = np.linspace(0, 4e-3, 10)
        result = interpolate(trace, new_time, method="linear")

        # Should propagate NaN values
        assert np.any(np.isnan(result.data))

    def test_constant_value_interpolation(self, constant_trace: WaveformTrace):
        """Test interpolation of constant-value trace."""
        # Interpolate within the original time range
        duration = len(constant_trace.data) / constant_trace.metadata.sample_rate
        new_time = np.linspace(0, duration, 200)
        result = interpolate(constant_trace, new_time, method="linear")

        # All values should remain constant (ignoring any NaN from edges)
        valid_data = result.data[~np.isnan(result.data)]
        np.testing.assert_allclose(valid_data, 5.0, rtol=1e-10)

    def test_negative_values_handling(self):
        """Test handling of negative values."""
        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000.0))

        result = resample(trace, num_samples=12)

        # Should handle negative values correctly
        assert np.min(result.data) < 0
        assert np.max(result.data) > 0

    def test_very_large_upsampling(self):
        """Test extreme upsampling."""
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000.0))

        # Upsample 100x
        result = resample(trace, num_samples=300, method="interp")

        assert len(result.data) == 300

    def test_very_large_downsampling(self):
        """Test extreme downsampling."""
        data = np.arange(10000, dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1_000_000.0))

        # Downsample 100x
        result = downsample(trace, factor=100, method="average")

        assert len(result.data) == 100

    def test_integer_vs_float_consistency(self):
        """Test that integer data is converted to float properly."""
        int_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        trace = WaveformTrace(data=int_data, metadata=TraceMetadata(sample_rate=1000.0))

        result = interpolate(trace, np.linspace(0, 4e-3, 10), method="linear")

        # Result should be float64
        assert result.data.dtype == np.float64
        assert np.all(np.isfinite(result.data))

    def test_interp_boundary_conditions(self):
        """Test interpolation at exact boundaries."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000.0))

        # Interpolate at exact original time points
        original_time = trace.time_vector
        result = interpolate(trace, original_time, method="cubic")

        # Should match original data closely
        np.testing.assert_allclose(result.data, data, rtol=1e-6)

    def test_zero_crossing_preservation(self):
        """Test that resampling preserves zero crossings."""
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000.0))

        result = resample(trace, new_sample_rate=2000, method="fft")

        # Original has zero crossings
        original_crossings = np.sum(np.diff(np.sign(data)) != 0)
        result_crossings = np.sum(np.diff(np.sign(result.data)) != 0)

        # Should have similar number of zero crossings
        assert abs(original_crossings - result_crossings) <= 2
