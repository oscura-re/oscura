"""Tests for Jupyter integration.

Tests requirements:
  - Magic commands
  - Rich display
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestMagicCommands:
    """Test IPython magic command functionality."""

    def test_magics_class_exists(self) -> None:
        """Test that OscuraMagics class exists."""
        from oscura.jupyter.magic import OscuraMagics

        assert OscuraMagics is not None

    def test_current_trace_management(self) -> None:
        """Test trace management functions."""
        from oscura.jupyter.magic import (
            get_current_trace,
            set_current_trace,
        )

        # Initially None
        get_current_trace()

        # Set a trace
        mock_trace = {"data": [1, 2, 3]}
        set_current_trace(mock_trace, "test.csv")

        # Should return it
        assert get_current_trace() == mock_trace

        # Clean up
        set_current_trace(None)


class TestDisplay:
    """Test rich display functionality."""

    def test_trace_display_repr_html(self) -> None:
        """Test TraceDisplay generates HTML."""
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.jupyter.display import TraceDisplay

        data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=1e6, channel="CH1"),
        )

        display = TraceDisplay(trace, title="Test Trace")
        html = display._repr_html_()

        assert "<html>" in html or "<div" in html
        assert "Test Trace" in html
        assert "1,000" in html or "Samples" in html

    def test_measurement_display_repr_html(self) -> None:
        """Test MeasurementDisplay generates HTML."""
        from oscura.jupyter.display import MeasurementDisplay

        measurements = {
            "rise_time": 2.5e-9,
            "frequency": 10e6,
            "amplitude": 3.3,
        }

        display = MeasurementDisplay(measurements, title="Test Measurements")
        html = display._repr_html_()

        assert "<html>" in html or "<div" in html
        assert "Test Measurements" in html

    def test_measurement_format_value(self) -> None:
        """Test value formatting in MeasurementDisplay."""
        from oscura.jupyter.display import MeasurementDisplay

        display = MeasurementDisplay({})

        # Nanoseconds
        result = display._format_value(2.5e-9)
        assert "n" in result  # nanoseconds

        # Megahertz
        result = display._format_value(10e6)
        assert "M" in result  # mega

        # Small values
        result = display._format_value(0)
        assert result == "0"


class TestDisplayFunctions:
    """Test display convenience functions."""

    def test_display_trace_function(self) -> None:
        """Test display_trace function."""
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.jupyter.display import display_trace

        data = np.sin(2 * np.pi * np.linspace(0, 1, 100))
        trace = WaveformTrace(data=data.astype(np.float64), metadata=TraceMetadata(sample_rate=1e6))

        # Should not raise
        display_trace(trace, title="Test")

    def test_display_measurements_function(self) -> None:
        """Test display_measurements function."""
        from oscura.jupyter.display import display_measurements

        measurements = {"test": 1.0}

        # Should not raise
        display_measurements(measurements, title="Test")
