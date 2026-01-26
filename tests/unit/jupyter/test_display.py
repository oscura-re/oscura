"""Comprehensive tests for Jupyter display module.

Tests requirements:
  - TraceDisplay HTML generation
  - MeasurementDisplay formatting
  - Display functions (trace, measurements, spectrum)
  - Fallback behavior when IPython not available
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.jupyter.display import (
    HTML,
    MeasurementDisplay,
    TraceDisplay,
    display_measurements,
    display_spectrum,
    display_trace,
)

pytestmark = pytest.mark.unit


class TestTraceDisplay:
    """Test TraceDisplay class for trace HTML rendering."""

    def test_init(self) -> None:
        """Test TraceDisplay initialization."""
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        display = TraceDisplay(trace, title="My Trace")

        assert display.trace == trace
        assert display.title == "My Trace"

    def test_repr_html_basic(self) -> None:
        """Test HTML generation with basic trace."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        display = TraceDisplay(trace)
        html = display._repr_html_()

        assert "<div" in html
        assert "Trace" in html
        assert "5" in html  # sample count

    def test_repr_html_with_metadata(self) -> None:
        """Test HTML generation with full metadata."""
        data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=1e9,
                channel_name="CH1",
                source_file="test.wfm",
            ),
        )

        display = TraceDisplay(trace, title="Test Signal")
        html = display._repr_html_()

        assert "Test Signal" in html
        assert "1,000" in html or "1000" in html
        assert "CH1" in html
        assert "test.wfm" in html
        assert "GSa/s" in html

    def test_repr_html_statistics(self) -> None:
        """Test that statistics are included in HTML."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        display = TraceDisplay(trace)
        html = display._repr_html_()

        assert "Min" in html
        assert "Max" in html
        assert "Mean" in html
        assert "Std Dev" in html

    def test_format_sample_rate_giga(self) -> None:
        """Test sample rate formatting in GHz range."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_sample_rate(2.5e9)
        assert "2.500 GSa/s" in result

    def test_format_sample_rate_mega(self) -> None:
        """Test sample rate formatting in MHz range."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_sample_rate(100e6)
        assert "100.000 MSa/s" in result

    def test_format_sample_rate_kilo(self) -> None:
        """Test sample rate formatting in kHz range."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_sample_rate(500e3)
        assert "500.000 kSa/s" in result

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting in seconds."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_duration(2.5)
        assert "2.500 s" in result

    def test_format_duration_milliseconds(self) -> None:
        """Test duration formatting in milliseconds."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_duration(0.005)
        assert "5.000 ms" in result

    def test_format_duration_microseconds(self) -> None:
        """Test duration formatting in microseconds."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_duration(5e-6)
        assert "5.000 us" in result

    def test_format_duration_nanoseconds(self) -> None:
        """Test duration formatting in nanoseconds."""
        trace = Mock()
        display = TraceDisplay(trace)

        result = display._format_duration(5e-9)
        assert "5.000 ns" in result

    def test_trace_without_metadata(self) -> None:
        """Test display of trace without metadata."""

        class SimpleTrace:
            """Minimal trace object."""

            def __init__(self) -> None:
                self.data = np.array([1.0, 2.0, 3.0])

        trace = SimpleTrace()
        display = TraceDisplay(trace)
        html = display._repr_html_()

        assert "<div" in html
        assert "3" in html  # sample count


class TestMeasurementDisplay:
    """Test MeasurementDisplay class for measurement formatting."""

    def test_init(self) -> None:
        """Test MeasurementDisplay initialization."""
        measurements = {"test": 1.0}
        display = MeasurementDisplay(measurements, title="Results")

        assert display.measurements == measurements
        assert display.title == "Results"

    def test_repr_html(self) -> None:
        """Test HTML generation for measurements."""
        measurements = {
            "rise_time": 2.5e-9,
            "frequency": 10e6,
            "amplitude": 3.3,
        }

        display = MeasurementDisplay(measurements, title="Test Results")
        html = display._repr_html_()

        assert "<div" in html
        assert "Test Results" in html
        assert "rise_time" in html
        assert "frequency" in html
        assert "amplitude" in html

    def test_format_value_zero(self) -> None:
        """Test formatting of zero values."""
        display = MeasurementDisplay({})
        result = display._format_value(0.0)
        assert result == "0"

    def test_format_value_giga(self) -> None:
        """Test formatting of gigascale values."""
        display = MeasurementDisplay({})
        result = display._format_value(5e9)
        assert "5.000 G" in result

    def test_format_value_mega(self) -> None:
        """Test formatting of megascale values."""
        display = MeasurementDisplay({})
        result = display._format_value(10e6)
        assert "10.000 M" in result

    def test_format_value_kilo(self) -> None:
        """Test formatting of kiloscale values."""
        display = MeasurementDisplay({})
        result = display._format_value(2.5e3)
        assert "2.500 k" in result

    def test_format_value_normal(self) -> None:
        """Test formatting of normal scale values."""
        display = MeasurementDisplay({})
        result = display._format_value(3.14159)
        assert "3.14" in result

    def test_format_value_milli(self) -> None:
        """Test formatting of milliscale values."""
        display = MeasurementDisplay({})
        result = display._format_value(0.005)
        assert "5.000 m" in result

    def test_format_value_micro(self) -> None:
        """Test formatting of microscale values."""
        display = MeasurementDisplay({})
        result = display._format_value(5e-6)
        assert "5.000 u" in result

    def test_format_value_nano(self) -> None:
        """Test formatting of nanoscale values."""
        display = MeasurementDisplay({})
        result = display._format_value(2.5e-9)
        assert "2.500 n" in result

    def test_format_value_pico(self) -> None:
        """Test formatting of picoscale values."""
        display = MeasurementDisplay({})
        result = display._format_value(1e-12)
        assert "1.000 p" in result

    def test_format_value_scientific(self) -> None:
        """Test formatting of very small values in scientific notation."""
        display = MeasurementDisplay({})
        result = display._format_value(1e-15)
        assert "e" in result.lower()

    def test_format_value_non_float(self) -> None:
        """Test formatting of non-float values."""
        display = MeasurementDisplay({})
        result = display._format_value("text")
        assert result == "text"

        result = display._format_value(42)
        assert result == "42"


class TestDisplayFunctions:
    """Test display convenience functions."""

    def test_display_trace(self) -> None:
        """Test display_trace function."""
        data = np.sin(2 * np.pi * np.linspace(0, 1, 100))
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        # Should not raise
        display_trace(trace, title="Test Trace")

    def test_display_measurements(self) -> None:
        """Test display_measurements function."""
        measurements = {
            "rise_time": 2.5e-9,
            "frequency": 10e6,
        }

        # Should not raise
        display_measurements(measurements, title="Test Measurements")

    @patch("matplotlib.pyplot.show")
    def test_display_spectrum(self, mock_show: Mock) -> None:
        """Test display_spectrum function with mock plot."""
        frequencies = np.linspace(0, 100e6, 1000)
        magnitudes = -20 * np.log10(frequencies / frequencies[1] + 1)

        # Should not raise
        display_spectrum(frequencies, magnitudes, title="Test Spectrum")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_display_spectrum_linear_scale(self, mock_show: Mock) -> None:
        """Test display_spectrum with linear scale."""
        frequencies = np.linspace(0, 100e6, 1000)
        magnitudes = np.ones_like(frequencies)

        display_spectrum(
            frequencies,
            magnitudes,
            title="Linear Spectrum",
            log_scale=False,
        )
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_display_spectrum_custom_figsize(self, mock_show: Mock) -> None:
        """Test display_spectrum with custom figure size."""
        frequencies = np.linspace(1, 100e6, 1000)
        magnitudes = np.ones_like(frequencies)

        display_spectrum(
            frequencies,
            magnitudes,
            figsize=(12, 6),
        )
        mock_show.assert_called_once()


class TestFallbackBehavior:
    """Test fallback behavior when IPython not available."""

    def test_fallback_html_class(self) -> None:
        """Test fallback HTML class."""
        # When IPython not available, should have fallback HTML class
        html = HTML("<div>Test</div>")
        assert html.data == "<div>Test</div>"

    def test_display_trace_without_ipython(self) -> None:
        """Test display_trace fallback when IPython not available."""
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        # Should not raise even without IPython
        with patch("oscura.jupyter.display.IPYTHON_AVAILABLE", False):
            display_trace(trace)

    def test_display_measurements_without_ipython(self) -> None:
        """Test display_measurements fallback when IPython not available."""
        measurements = {"test": 1.0}

        # Should not raise even without IPython
        with patch("oscura.jupyter.display.IPYTHON_AVAILABLE", False):
            display_measurements(measurements)
