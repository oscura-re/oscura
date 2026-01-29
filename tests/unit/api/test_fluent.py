"""Comprehensive unit tests for fluent API module.


Test coverage:
- FluentResult: value wrapping, map, filter, format, print, metadata
- FluentTrace: initialization, data access, method chaining
- Filtering: lowpass, highpass, bandpass, notch
- Transforms: normalize, scale, offset, clip, abs, diff, integrate
- Resampling: resample, decimate, slice
- Spectral: fft, magnitude, phase, psd
- Measurements: mean, std, rms, peak_to_peak, min, max
- Utilities: copy, history, metadata, trace() factory
- Edge cases and error handling
"""

import numpy as np
import pytest

from oscura.api.fluent import FluentResult, FluentTrace, trace

pytestmark = pytest.mark.unit


# =============================================================================
# FluentResult Tests
# =============================================================================


class TestFluentResult:
    """Test FluentResult class (API-019)."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        result = FluentResult(42)
        assert result.value == 42
        assert result.metadata == {}

    def test_init_with_metadata(self) -> None:
        """Test initialization with metadata."""
        result = FluentResult(42.5, {"unit": "Hz", "source": "test"})
        assert result.value == 42.5
        assert result.metadata == {"unit": "Hz", "source": "test"}

    def test_get(self) -> None:
        """Test get() returns wrapped value."""
        result = FluentResult([1, 2, 3])
        assert result.get() == [1, 2, 3]

    def test_map(self) -> None:
        """Test map() applies function to value."""
        result = FluentResult(10)
        mapped = result.map(lambda x: x * 2)

        assert mapped.value == 20
        assert isinstance(mapped, FluentResult)

    def test_map_preserves_metadata(self) -> None:
        """Test map() preserves metadata."""
        result = FluentResult(10, {"unit": "V"})
        mapped = result.map(lambda x: x * 2)

        assert mapped.metadata == {"unit": "V"}

    def test_map_copies_metadata(self) -> None:
        """Test map() copies metadata (not reference)."""
        result = FluentResult(10, {"count": 1})
        mapped = result.map(lambda x: x * 2)

        # Modify original metadata
        result.metadata["count"] = 2

        # Mapped should have original value
        assert mapped.metadata["count"] == 1

    def test_filter_pass(self) -> None:
        """Test filter() passes when predicate is true."""
        result = FluentResult(10)
        filtered = result.filter(lambda x: x > 5)

        assert filtered is result

    def test_filter_fail(self) -> None:
        """Test filter() returns None when predicate is false."""
        result = FluentResult(10)
        filtered = result.filter(lambda x: x > 20)

        assert filtered is None

    def test_format(self) -> None:
        """Test format() creates formatted string."""
        result = FluentResult(42.567)
        formatted = result.format("Value: {:.2f}")

        assert formatted.value == "Value: 42.57"
        assert isinstance(formatted, FluentResult)

    def test_format_preserves_metadata(self) -> None:
        """Test format() preserves metadata."""
        result = FluentResult(42.567, {"unit": "Hz"})
        formatted = result.format("Value: {:.2f}")

        assert formatted.metadata == {"unit": "Hz"}

    def test_print(self, capsys) -> None:
        """Test print() outputs value."""
        result = FluentResult(42)
        returned = result.print()

        captured = capsys.readouterr()
        assert captured.out.strip() == "42"
        assert returned is result

    def test_print_with_prefix(self, capsys) -> None:
        """Test print() with prefix."""
        result = FluentResult(42)
        result.print("Result: ")

        captured = capsys.readouterr()
        assert captured.out.strip() == "Result: 42"

    def test_with_metadata(self) -> None:
        """Test with_metadata() adds metadata."""
        result = FluentResult(42)
        returned = result.with_metadata(unit="Hz", source="test")

        assert returned is result
        assert result.metadata == {"unit": "Hz", "source": "test"}

    def test_with_metadata_updates(self) -> None:
        """Test with_metadata() updates existing metadata."""
        result = FluentResult(42, {"unit": "Hz"})
        result.with_metadata(source="test", unit="kHz")

        assert result.metadata == {"unit": "kHz", "source": "test"}

    def test_repr(self) -> None:
        """Test __repr__ output."""
        result = FluentResult(42)
        assert repr(result) == "FluentResult(42)"

    def test_chaining_map_filter_format(self) -> None:
        """Test chaining map, filter, and format."""
        result = FluentResult(10)
        final = result.map(lambda x: x * 2).filter(lambda x: x > 15).format("Result: {}")

        assert final is not None
        assert final.value == "Result: 20"


# =============================================================================
# FluentTrace Initialization Tests
# =============================================================================


class TestFluentTraceInit:
    """Test FluentTrace initialization."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)

        assert np.array_equal(trace.data, data)
        assert trace.sample_rate == 1.0
        assert trace._metadata == {}
        assert trace._history == []

    def test_init_with_sample_rate(self) -> None:
        """Test initialization with sample rate."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, sample_rate=1e9)

        assert trace.sample_rate == 1e9

    def test_init_with_metadata(self) -> None:
        """Test initialization with metadata."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, sample_rate=1e6, unit="V", source="scope")

        assert trace._metadata == {"unit": "V", "source": "scope"}

    def test_data_property(self) -> None:
        """Test data property accessor."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)

        assert np.array_equal(trace.data, data)

    def test_sample_rate_property(self) -> None:
        """Test sample_rate property accessor."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, sample_rate=1e9)

        assert trace.sample_rate == 1e9

    def test_get(self) -> None:
        """Test get() returns raw data."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)

        assert np.array_equal(trace.get(), data)

    def test_copy(self) -> None:
        """Test copy() creates independent copy."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, sample_rate=1e6, unit="V")
        copy = trace.copy()

        # Should have same values
        assert np.array_equal(copy.data, trace.data)
        assert copy.sample_rate == trace.sample_rate
        assert copy._metadata == trace._metadata

        # But be independent
        copy.data[0] = 999
        assert trace.data[0] == 1.0

    def test_len(self) -> None:
        """Test __len__ returns data length."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = FluentTrace(data)

        assert len(trace) == 5

    def test_repr(self) -> None:
        """Test __repr__ output."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, sample_rate=1e9)

        repr_str = repr(trace)
        assert "FluentTrace" in repr_str
        assert "samples=3" in repr_str
        assert "sample_rate=1000000000.0" in repr_str
        assert "operations=0" in repr_str


# =============================================================================
# Filtering Methods Tests
# =============================================================================


class TestFluentTraceFiltering:
    """Test filtering methods."""

    def test_lowpass(self) -> None:
        """Test lowpass filter."""
        # Create signal with high frequency noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

        trace = FluentTrace(signal, sample_rate=1000)
        filtered = trace.lowpass(cutoff=50)

        # Should remove high frequency component
        assert filtered is trace  # Chaining
        assert "lowpass(cutoff=50)" in trace._history
        # Signal should be smoother (less variance after filtering)
        assert np.std(trace.data) < np.std(signal)

    def test_lowpass_order(self) -> None:
        """Test lowpass filter with custom order."""
        data = np.random.randn(1000)
        trace = FluentTrace(data, sample_rate=1000)
        trace.lowpass(cutoff=100, order=8)

        assert "lowpass(cutoff=100)" in trace._history

    def test_highpass(self) -> None:
        """Test highpass filter."""
        # Create signal with DC offset and low frequency
        t = np.linspace(0, 1, 1000)
        signal = 5.0 + np.sin(2 * np.pi * 10 * t)

        trace = FluentTrace(signal, sample_rate=1000)
        filtered = trace.highpass(cutoff=5)

        assert filtered is trace
        assert "highpass(cutoff=5)" in trace._history
        # Should remove DC component
        assert abs(np.mean(trace.data)) < abs(np.mean(signal))

    def test_bandpass(self) -> None:
        """Test bandpass filter."""
        # Create signal with multiple frequencies
        t = np.linspace(0, 1, 1000)
        signal = (
            np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        )

        trace = FluentTrace(signal, sample_rate=1000)
        filtered = trace.bandpass(low=30, high=70)

        assert filtered is trace
        assert "bandpass(low=30, high=70)" in trace._history

    def test_notch(self) -> None:
        """Test notch filter."""
        # Create signal with 50Hz noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        trace = FluentTrace(signal, sample_rate=1000)
        filtered = trace.notch(freq=50, Q=30)

        assert filtered is trace
        assert "notch(freq=50)" in trace._history


# =============================================================================
# Transform Methods Tests
# =============================================================================


class TestFluentTraceTransforms:
    """Test transform methods."""

    def test_normalize_minmax(self) -> None:
        """Test minmax normalization."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        trace = FluentTrace(data)
        normalized = trace.normalize(method="minmax")

        assert normalized is trace
        assert trace.data[0] == 0.0
        assert trace.data[-1] == 1.0
        assert "normalize(method=minmax)" in trace._history

    def test_normalize_minmax_constant(self) -> None:
        """Test minmax normalization on constant data."""
        data = np.array([5.0, 5.0, 5.0])
        trace = FluentTrace(data)
        trace.normalize(method="minmax")

        # Should not crash on constant data
        assert len(trace.data) == 3

    def test_normalize_zscore(self) -> None:
        """Test z-score normalization."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        trace = FluentTrace(data)
        trace.normalize(method="zscore")

        assert abs(np.mean(trace.data)) < 1e-10
        assert abs(np.std(trace.data) - 1.0) < 1e-10
        assert "normalize(method=zscore)" in trace._history

    def test_normalize_zscore_constant(self) -> None:
        """Test z-score normalization on constant data."""
        data = np.array([5.0, 5.0, 5.0])
        trace = FluentTrace(data)
        trace.normalize(method="zscore")

        # Should not crash on zero std
        assert len(trace.data) == 3

    def test_normalize_peak(self) -> None:
        """Test peak normalization."""
        data = np.array([-10.0, 5.0, 20.0, -15.0])
        trace = FluentTrace(data)
        trace.normalize(method="peak")

        assert np.max(np.abs(trace.data)) == 1.0
        assert "normalize(method=peak)" in trace._history

    def test_normalize_peak_zero(self) -> None:
        """Test peak normalization on zero data."""
        data = np.array([0.0, 0.0, 0.0])
        trace = FluentTrace(data)
        trace.normalize(method="peak")

        # Should not crash on zero peak
        assert len(trace.data) == 3

    def test_normalize_unknown_method(self) -> None:
        """Test normalize with unknown method (should be no-op)."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)
        original = data.copy()
        trace.normalize(method="unknown")

        # Data should be unchanged (unknown method is a no-op)
        assert np.array_equal(trace.data, original)
        assert "normalize(method=unknown)" in trace._history

    def test_scale(self) -> None:
        """Test scale operation."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)
        scaled = trace.scale(2.5)

        assert scaled is trace
        assert np.allclose(trace.data, [2.5, 5.0, 7.5])
        assert "scale(factor=2.5)" in trace._history

    def test_offset(self) -> None:
        """Test offset operation."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)
        offset = trace.offset(10.0)

        assert offset is trace
        assert np.allclose(trace.data, [11.0, 12.0, 13.0])
        assert "offset(value=10.0)" in trace._history

    def test_clip(self) -> None:
        """Test clip operation."""
        data = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        trace = FluentTrace(data)
        clipped = trace.clip(low=0.0, high=10.0)

        assert clipped is trace
        assert np.allclose(trace.data, [0.0, 0.0, 5.0, 10.0, 10.0])
        assert "clip(low=0.0, high=10.0)" in trace._history

    def test_abs(self) -> None:
        """Test absolute value."""
        data = np.array([-5.0, -2.0, 0.0, 3.0, -7.0])
        trace = FluentTrace(data)
        abs_trace = trace.abs()

        assert abs_trace is trace
        assert np.allclose(trace.data, [5.0, 2.0, 0.0, 3.0, 7.0])
        assert "abs()" in trace._history

    def test_diff(self) -> None:
        """Test differentiation."""
        data = np.array([1.0, 3.0, 6.0, 10.0])
        trace = FluentTrace(data)
        diff_trace = trace.diff()

        assert diff_trace is trace
        assert len(trace.data) == 4  # Same length due to prepend
        assert np.allclose(trace.data[1:], [2.0, 3.0, 4.0])
        assert "diff()" in trace._history

    def test_integrate(self) -> None:
        """Test integration."""
        data = np.array([1.0, 1.0, 1.0, 1.0])
        trace = FluentTrace(data, sample_rate=1.0)
        integrated = trace.integrate()

        assert integrated is trace
        assert np.allclose(trace.data, [1.0, 2.0, 3.0, 4.0])
        assert "integrate()" in trace._history


# =============================================================================
# Resampling Methods Tests
# =============================================================================


class TestFluentTraceResampling:
    """Test resampling methods."""

    def test_resample(self) -> None:
        """Test resampling to new length."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = FluentTrace(data, sample_rate=1000)
        resampled = trace.resample(new_length=10)

        assert resampled is trace
        assert len(trace.data) == 10
        assert "resample(new_length=10)" in trace._history

    def test_decimate(self) -> None:
        """Test decimation."""
        data = np.arange(100, dtype=float)
        trace = FluentTrace(data, sample_rate=1000)
        decimated = trace.decimate(factor=5)

        assert decimated is trace
        assert len(trace.data) == 20
        assert trace.sample_rate == 200
        assert "decimate(factor=5)" in trace._history

    def test_slice(self) -> None:
        """Test slicing."""
        data = np.arange(100, dtype=float)
        trace = FluentTrace(data)
        sliced = trace.slice(start=10, end=20)

        assert sliced is trace
        assert len(trace.data) == 10
        assert np.allclose(trace.data, np.arange(10, 20))
        assert "slice(start=10, end=20)" in trace._history

    def test_slice_no_end(self) -> None:
        """Test slicing without end."""
        data = np.arange(100, dtype=float)
        trace = FluentTrace(data)
        sliced = trace.slice(start=90)

        assert len(trace.data) == 10
        assert "slice(start=90, end=None)" in trace._history


# =============================================================================
# Spectral Methods Tests
# =============================================================================


class TestFluentTraceSpectral:
    """Test spectral methods."""

    def test_fft(self) -> None:
        """Test FFT computation."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
        trace = FluentTrace(data)
        fft_trace = trace.fft()

        assert fft_trace is trace
        assert len(trace.data) == 100
        assert np.iscomplexobj(trace.data)
        assert "fft(nfft=None)" in trace._history

    def test_fft_with_nfft(self) -> None:
        """Test FFT with custom size."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
        trace = FluentTrace(data)
        trace.fft(nfft=128)

        assert len(trace.data) == 128
        assert "fft(nfft=128)" in trace._history

    def test_magnitude(self) -> None:
        """Test magnitude of complex data."""
        data = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        trace = FluentTrace(data)
        mag = trace.magnitude()

        assert mag is trace
        assert np.allclose(trace.data, [np.sqrt(2), np.sqrt(8), np.sqrt(18)])
        assert "magnitude()" in trace._history

    def test_phase(self) -> None:
        """Test phase of complex data."""
        data = np.array([1 + 0j, 0 + 1j, -1 + 0j])
        trace = FluentTrace(data)
        phase = trace.phase()

        assert phase is trace
        assert np.allclose(trace.data, [0, np.pi / 2, np.pi])
        assert "phase()" in trace._history

    def test_psd(self) -> None:
        """Test power spectral density."""
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 50 * t)
        trace = FluentTrace(data, sample_rate=1000)

        result = trace.psd(nperseg=256)

        assert isinstance(result, FluentResult)
        freq, psd = result.value
        assert len(freq) > 0
        assert len(psd) > 0
        assert len(freq) == len(psd)


# =============================================================================
# Measurement Methods Tests
# =============================================================================


class TestFluentTraceMeasurements:
    """Test measurement methods."""

    def test_mean(self) -> None:
        """Test mean calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = FluentTrace(data)
        result = trace.mean()

        assert isinstance(result, FluentResult)
        assert result.value == 3.0

    def test_std(self) -> None:
        """Test standard deviation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = FluentTrace(data)
        result = trace.std()

        assert isinstance(result, FluentResult)
        assert result.value == pytest.approx(np.std(data))

    def test_rms(self) -> None:
        """Test RMS calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = FluentTrace(data)
        result = trace.rms()

        assert isinstance(result, FluentResult)
        expected_rms = np.sqrt(np.mean(data**2))
        assert result.value == pytest.approx(expected_rms)

    def test_peak_to_peak(self) -> None:
        """Test peak-to-peak calculation."""
        data = np.array([-5.0, -2.0, 0.0, 3.0, 7.0])
        trace = FluentTrace(data)
        result = trace.peak_to_peak()

        assert isinstance(result, FluentResult)
        assert result.value == 12.0

    def test_min(self) -> None:
        """Test minimum value."""
        data = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        trace = FluentTrace(data)
        result = trace.min()

        assert isinstance(result, FluentResult)
        assert result.value == 1.0

    def test_max(self) -> None:
        """Test maximum value."""
        data = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        trace = FluentTrace(data)
        result = trace.max()

        assert isinstance(result, FluentResult)
        assert result.value == 9.0


# =============================================================================
# Utility Methods Tests
# =============================================================================


class TestFluentTraceUtilities:
    """Test utility methods."""

    def test_print_history(self, capsys) -> None:
        """Test printing operation history."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)
        trace.scale(2).offset(1).abs()

        returned = trace.print_history()

        assert returned is trace
        captured = capsys.readouterr()
        assert "Operation history:" in captured.out
        assert "scale(factor=2)" in captured.out
        assert "offset(value=1)" in captured.out
        assert "abs()" in captured.out

    def test_with_metadata(self) -> None:
        """Test adding metadata."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data)
        returned = trace.with_metadata(unit="V", source="scope")

        assert returned is trace
        assert trace._metadata == {"unit": "V", "source": "scope"}

    def test_with_metadata_update(self) -> None:
        """Test updating existing metadata."""
        data = np.array([1.0, 2.0, 3.0])
        trace = FluentTrace(data, unit="V")
        trace.with_metadata(source="scope", unit="mV")

        assert trace._metadata == {"unit": "mV", "source": "scope"}


# =============================================================================
# Method Chaining Tests
# =============================================================================


class TestMethodChaining:
    """Test method chaining workflows."""

    def test_filter_chain(self) -> None:
        """Test chaining multiple filters."""
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

        trace = (
            FluentTrace(data, sample_rate=1000).lowpass(cutoff=50).highpass(cutoff=5).normalize()
        )

        assert len(trace._history) == 3
        assert "lowpass" in trace._history[0]
        assert "highpass" in trace._history[1]
        assert "normalize" in trace._history[2]

    def test_transform_chain(self) -> None:
        """Test chaining transforms."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        trace = FluentTrace(data).scale(2).offset(10).clip(low=0, high=20).normalize()

        assert len(trace._history) == 4

    def test_spectral_chain(self) -> None:
        """Test spectral analysis chain."""
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 10 * t)

        result = FluentTrace(data, sample_rate=100).normalize().fft().magnitude().max()

        assert isinstance(result, FluentResult)
        assert result.value > 0

    def test_complex_workflow(self) -> None:
        """Test complex workflow with filtering, transform, and measurement."""
        t = np.linspace(0, 1, 1000)
        data = 5.0 + np.sin(2 * np.pi * 10 * t) + 0.3 * np.random.randn(1000)

        result = (
            FluentTrace(data, sample_rate=1000)
            .highpass(cutoff=5)
            .lowpass(cutoff=50)
            .normalize(method="zscore")
            .abs()
            .mean()
        )

        assert isinstance(result, FluentResult)
        assert isinstance(result.value, float)

    def test_measurement_chain(self) -> None:
        """Test chaining measurements via FluentResult."""
        data = np.array([10.0, 20.0, 30.0])

        result = FluentTrace(data).mean().map(lambda x: x * 2).format("Mean: {:.1f}").get()

        assert result == "Mean: 40.0"

    def test_history_tracking(self) -> None:
        """Test operation history is correctly tracked."""
        data = np.arange(100, dtype=float)

        trace = (
            FluentTrace(data, sample_rate=1000)
            .lowpass(cutoff=100)
            .normalize()
            .scale(2.5)
            .slice(start=10, end=90)
        )

        assert len(trace._history) == 4
        assert all(isinstance(h, str) for h in trace._history)


# =============================================================================
# trace() Factory Function Tests
# =============================================================================


class TestTraceFactory:
    """Test trace() factory function."""

    def test_trace_basic(self) -> None:
        """Test basic trace creation."""
        data = np.array([1.0, 2.0, 3.0])
        t = trace(data)

        assert isinstance(t, FluentTrace)
        assert np.array_equal(t.data, data)
        assert t.sample_rate == 1.0

    def test_trace_with_sample_rate(self) -> None:
        """Test trace with sample rate."""
        data = np.array([1.0, 2.0, 3.0])
        t = trace(data, sample_rate=1e9)

        assert t.sample_rate == 1e9

    def test_trace_with_metadata(self) -> None:
        """Test trace with metadata."""
        data = np.array([1.0, 2.0, 3.0])
        t = trace(data, sample_rate=1e6, unit="V", source="scope")

        assert t._metadata == {"unit": "V", "source": "scope"}

    def test_trace_chaining(self) -> None:
        """Test factory function with immediate chaining."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = trace(data).scale(2).offset(10).mean().get()

        expected = np.mean([12, 14, 16, 18, 20])
        assert result == pytest.approx(expected)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestApiFluentEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self) -> None:
        """Test with empty array."""
        data = np.array([])
        trace = FluentTrace(data)

        assert len(trace) == 0

    def test_single_sample(self) -> None:
        """Test with single sample."""
        data = np.array([42.0])
        trace = FluentTrace(data)

        assert len(trace) == 1
        assert trace.mean().value == 42.0

    def test_very_large_data(self) -> None:
        """Test with large array."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000000)
        trace = FluentTrace(data)

        # After zscore normalization, mean should be near zero
        result = trace.normalize(method="zscore").mean().value
        assert abs(result) < 1e-10  # Should be essentially zero after zscore

    def test_complex_data_handling(self) -> None:
        """Test operations on complex data."""
        data = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        trace = FluentTrace(data)

        # Magnitude should work
        trace.magnitude()
        assert np.all(np.isreal(trace.data))

    def test_nan_handling(self) -> None:
        """Test with NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        trace = FluentTrace(data)

        # Operations should handle NaN gracefully
        result = trace.mean().value
        assert np.isnan(result)

    def test_inf_handling(self) -> None:
        """Test with infinite values."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        trace = FluentTrace(data)

        result = trace.max().value
        assert np.isinf(result)

    def test_filter_edge_case_nyquist(self) -> None:
        """Test filter at Nyquist frequency."""
        data = np.random.randn(1000)
        trace = FluentTrace(data, sample_rate=1000)

        # Cutoff at 99% of Nyquist should not crash
        trace.lowpass(cutoff=499)
        assert len(trace.data) == 1000

    def test_resample_to_smaller(self) -> None:
        """Test resampling to smaller size."""
        data = np.arange(1000, dtype=float)
        trace = FluentTrace(data)
        trace.resample(new_length=100)

        assert len(trace.data) == 100

    def test_resample_to_larger(self) -> None:
        """Test resampling to larger size."""
        data = np.arange(100, dtype=float)
        trace = FluentTrace(data)
        trace.resample(new_length=1000)

        assert len(trace.data) == 1000

    def test_slice_out_of_bounds(self) -> None:
        """Test slicing beyond data bounds."""
        data = np.arange(100, dtype=float)
        trace = FluentTrace(data)
        trace.slice(start=50, end=200)

        # Should handle gracefully
        assert len(trace.data) == 50


# =============================================================================
# Integration Tests
# =============================================================================


class TestApiFluentIntegration:
    """Integration tests for realistic workflows."""

    def test_signal_processing_workflow(self) -> None:
        """Test complete signal processing workflow."""
        # Generate noisy signal
        t = np.linspace(0, 1, 10000)
        signal = 2.0 + 1.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.random.randn(10000)

        # Process and measure
        result = (
            trace(signal, sample_rate=10000)
            .highpass(cutoff=10)  # Remove DC
            .lowpass(cutoff=100)  # Remove high freq noise
            .normalize(method="zscore")
            .abs()
            .mean()
            .format("Processed mean: {:.4f}")
            .get()
        )

        assert "Processed mean:" in result
        assert isinstance(result, str)

    def test_spectral_analysis_workflow(self) -> None:
        """Test spectral analysis workflow."""
        # Generate signal with known frequency
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t)

        freq, psd = trace(signal, sample_rate=1000).normalize().psd(nperseg=256).get()

        # Peak should be near 50 Hz
        peak_idx = np.argmax(psd)
        peak_freq = freq[peak_idx]
        assert 45 <= peak_freq <= 55

    def test_multi_trace_comparison(self) -> None:
        """Test comparing multiple traces."""
        rng = np.random.default_rng(42)
        data1 = rng.standard_normal(1000)
        data2 = rng.standard_normal(1000)

        # Use zscore normalization which guarantees mean=0
        mean1 = trace(data1).normalize(method="zscore").mean().value
        mean2 = trace(data2).normalize(method="zscore").mean().value

        # After zscore normalization, mean should be essentially zero
        assert abs(mean1) < 1e-10
        assert abs(mean2) < 1e-10

    def test_metadata_preservation(self) -> None:
        """Test metadata is preserved through operations."""
        data = np.random.randn(100)

        t = (
            trace(data, sample_rate=1e6, unit="V", source="scope1")
            .lowpass(cutoff=1e5)
            .normalize()
            .scale(2.0)
        )

        assert t._metadata["unit"] == "V"
        assert t._metadata["source"] == "scope1"
        assert t.sample_rate == 1e6

    def test_copy_independence(self) -> None:
        """Test that copies are independent."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        original = trace(data)
        copy = original.copy()

        # Modify copy
        copy.scale(10)

        # Original should be unchanged
        assert np.array_equal(original.data, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(copy.data, [10.0, 20.0, 30.0, 40.0, 50.0])
