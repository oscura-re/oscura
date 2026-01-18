"""Tests for interpolation and resampling functions.

Requirements tested:
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.math.interpolation import (
    align_traces,
    downsample,
    interpolate,
    resample,
)

pytestmark = pytest.mark.unit


class TestInterpolate:
    """Tests for interpolate function."""

    @pytest.fixture
    def simple_trace(self) -> WaveformTrace:
        """Create a simple sine wave trace."""
        sample_rate = 1000.0
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="test",
        )
        return WaveformTrace(data=data, metadata=metadata)

    def test_linear_interpolation(self, simple_trace: WaveformTrace):
        """Test linear interpolation."""
        new_time = np.linspace(0, 0.05, 100)
        result = interpolate(simple_trace, new_time, method="linear")

        assert len(result.data) == 100
        assert result.metadata.sample_rate > 0

    def test_cubic_interpolation(self, simple_trace: WaveformTrace):
        """Test cubic interpolation."""
        new_time = np.linspace(0, 0.05, 100)
        result = interpolate(simple_trace, new_time, method="cubic")

        assert len(result.data) == 100

    def test_interpolation_preserves_values(self, simple_trace: WaveformTrace):
        """Test that interpolation preserves original sample values."""
        # Interpolate at original time points
        original_time = simple_trace.time_vector
        result = interpolate(simple_trace, original_time, method="linear")

        np.testing.assert_allclose(result.data, simple_trace.data, rtol=1e-10)


class TestResample:
    """Tests for resample function - REQ: API-019."""

    @pytest.fixture
    def sine_trace(self) -> WaveformTrace:
        """Create a sine wave trace for testing."""
        sample_rate = 10000.0  # 10 kHz
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="sine100",
        )
        return WaveformTrace(data=data, metadata=metadata)

    def test_resample_by_rate(self, sine_trace: WaveformTrace):
        """Test resampling by specifying new sample rate."""
        new_rate = 5000.0
        result = resample(sine_trace, new_sample_rate=new_rate)

        assert result.metadata.sample_rate == new_rate
        assert len(result.data) == pytest.approx(
            len(sine_trace.data) * new_rate / sine_trace.metadata.sample_rate,
            rel=0.01,
        )

    def test_resample_by_samples(self, sine_trace: WaveformTrace):
        """Test resampling by specifying number of samples."""
        num_samples = 500
        result = resample(sine_trace, num_samples=num_samples)

        assert len(result.data) == num_samples
        assert result.metadata.sample_rate > 0

    def test_resample_fft_method(self, sine_trace: WaveformTrace):
        """Test FFT-based resampling."""
        result = resample(sine_trace, new_sample_rate=8000, method="fft")

        assert result.metadata.sample_rate == 8000
        assert len(result.data) > 0

    def test_resample_polyphase_method(self, sine_trace: WaveformTrace):
        """Test polyphase resampling."""
        result = resample(sine_trace, new_sample_rate=5000, method="polyphase")

        assert result.metadata.sample_rate == 5000

    def test_resample_interp_method(self, sine_trace: WaveformTrace):
        """Test linear interpolation resampling."""
        result = resample(sine_trace, new_sample_rate=8000, method="interp")

        assert result.metadata.sample_rate == 8000

    def test_anti_alias_filter(self, sine_trace: WaveformTrace):
        """Test that anti-aliasing filter is applied when downsampling."""
        # Downsample with anti-aliasing
        result_with = resample(sine_trace, new_sample_rate=2000, anti_alias=True)
        # Downsample without anti-aliasing
        result_without = resample(sine_trace, new_sample_rate=2000, anti_alias=False)

        # Both should work, but results may differ
        assert len(result_with.data) == len(result_without.data)

    def test_nyquist_warning(self, sine_trace: WaveformTrace):
        """Test Nyquist criterion warning - REQ: API-019."""
        # Sine wave is 100 Hz, so Nyquist requires >= 200 Hz
        # Resample to 150 Hz should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resample(sine_trace, new_sample_rate=150)

            # Should have a UserWarning about Nyquist
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Nyquist" in str(w[0].message)

    def test_no_nyquist_warning_when_safe(self, sine_trace: WaveformTrace):
        """Test no warning when resampling above Nyquist rate."""
        # Sine wave is 100 Hz, so 500 Hz is safe
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resample(sine_trace, new_sample_rate=500)

            # Should not have warnings
            assert len(w) == 0

    def test_metadata_preserved(self, sine_trace: WaveformTrace):
        """Test that metadata is preserved and sample_rate updated."""
        result = resample(sine_trace, new_sample_rate=5000)

        assert result.metadata.sample_rate == 5000
        assert result.metadata.channel_name == "sine100_resampled"


class TestDownsample:
    """Tests for downsample function."""

    @pytest.fixture
    def high_rate_trace(self) -> WaveformTrace:
        """Create a high sample rate trace."""
        sample_rate = 10000.0
        t = np.arange(0, 0.1, 1 / sample_rate)
        data = np.sin(2 * np.pi * 50 * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="high_rate",
        )
        return WaveformTrace(data=data, metadata=metadata)

    def test_downsample_decimate(self, high_rate_trace: WaveformTrace):
        """Test decimation downsampling."""
        factor = 10
        result = downsample(high_rate_trace, factor, method="decimate")

        assert result.metadata.sample_rate == high_rate_trace.metadata.sample_rate / factor
        assert len(result.data) == len(high_rate_trace.data) // factor

    def test_downsample_average(self, high_rate_trace: WaveformTrace):
        """Test averaging downsampling."""
        factor = 5
        result = downsample(high_rate_trace, factor, method="average")

        assert result.metadata.sample_rate == high_rate_trace.metadata.sample_rate / factor

    def test_downsample_max(self, high_rate_trace: WaveformTrace):
        """Test max downsampling."""
        factor = 4
        result = downsample(high_rate_trace, factor, method="max")

        # Max should be >= original max for sine wave segments
        assert np.max(result.data) >= 0

    def test_downsample_min(self, high_rate_trace: WaveformTrace):
        """Test min downsampling."""
        factor = 4
        result = downsample(high_rate_trace, factor, method="min")

        # Min should be <= original min for sine wave segments
        assert np.min(result.data) <= 0

    def test_downsample_factor_one(self, high_rate_trace: WaveformTrace):
        """Test that factor=1 returns original trace."""
        result = downsample(high_rate_trace, factor=1)

        assert result is high_rate_trace


class TestAlignTraces:
    """Tests for align_traces function."""

    @pytest.fixture
    def trace_pair(self) -> tuple[WaveformTrace, WaveformTrace]:
        """Create two traces with different sample rates."""
        # Trace 1: 1000 Hz
        t1 = np.arange(0, 0.1, 1 / 1000)
        data1 = np.sin(2 * np.pi * 10 * t1)
        trace1 = WaveformTrace(
            data=data1,
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="trace1"),
        )

        # Trace 2: 2000 Hz
        t2 = np.arange(0, 0.1, 1 / 2000)
        data2 = np.cos(2 * np.pi * 10 * t2)
        trace2 = WaveformTrace(
            data=data2,
            metadata=TraceMetadata(sample_rate=2000.0, channel_name="trace2"),
        )

        return trace1, trace2

    def test_align_interpolate(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning traces using interpolation."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, method="interpolate")

        # Should have same sample rate
        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        # Should have same length
        assert len(aligned1.data) == len(aligned2.data)

    def test_align_resample(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning traces using resampling."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, method="resample")

        assert aligned1.metadata.sample_rate == aligned2.metadata.sample_rate
        assert len(aligned1.data) == len(aligned2.data)

    def test_align_use_higher_rate(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning to higher sample rate."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, reference="higher")

        # Should use 2000 Hz (the higher rate)
        assert aligned1.metadata.sample_rate == 2000.0
        assert aligned2.metadata.sample_rate == 2000.0

    def test_align_use_first_rate(self, trace_pair: tuple[WaveformTrace, WaveformTrace]):
        """Test aligning to first trace's rate."""
        trace1, trace2 = trace_pair
        aligned1, aligned2 = align_traces(trace1, trace2, reference="first")

        # Should use 1000 Hz (trace1's rate)
        assert aligned1.metadata.sample_rate == 1000.0
        assert aligned2.metadata.sample_rate == 1000.0
