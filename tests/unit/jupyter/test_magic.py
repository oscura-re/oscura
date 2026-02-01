"""Comprehensive tests for Jupyter magic commands.

Tests requirements:
  - OscuraMagics line and cell magics
  - Trace loading and management
  - Measurement execution
  - Help and info commands
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.jupyter.magic import (
    OscuraMagics,
    get_current_trace,
    load_ipython_extension,
    set_current_trace,
    unload_ipython_extension,
)

pytestmark = pytest.mark.unit


class TestTraceManagement:
    """Test current trace management functions."""

    def test_get_current_trace_initially_none(self) -> None:
        """Test that current trace is initially None."""
        # Reset to None first
        set_current_trace(None)
        result = get_current_trace()
        assert result is None

    def test_set_and_get_current_trace(self) -> None:
        """Test setting and getting current trace."""
        mock_trace = {"data": [1, 2, 3]}
        set_current_trace(mock_trace, "test.csv")

        result = get_current_trace()
        assert result == mock_trace

        # Cleanup
        set_current_trace(None)

    def test_set_current_trace_without_filename(self) -> None:
        """Test setting trace without filename."""
        mock_trace = {"data": [1, 2, 3]}
        set_current_trace(mock_trace)

        result = get_current_trace()
        assert result == mock_trace

        # Cleanup
        set_current_trace(None)

    def test_set_current_trace_overwrites(self) -> None:
        """Test that setting trace overwrites previous."""
        trace1 = {"data": [1, 2, 3]}
        trace2 = {"data": [4, 5, 6]}

        set_current_trace(trace1, "file1.csv")
        set_current_trace(trace2, "file2.csv")

        result = get_current_trace()
        assert result == trace2

        # Cleanup
        set_current_trace(None)


class TestOscuraMagicsClass:
    """Test OscuraMagics class."""

    def test_class_exists(self) -> None:
        """Test that OscuraMagics class can be instantiated."""
        magics = OscuraMagics()
        assert magics is not None

    def test_oscura_magic_help(self) -> None:
        """Test %oscura help command."""
        magics = OscuraMagics()

        # Should display help text
        with patch("builtins.print") as mock_print:
            magics.oscura("help")
            mock_print.assert_called()
            # Check that help text was printed
            call_args = str(mock_print.call_args)
            assert "Magic Commands" in call_args or "help" in call_args.lower()

    def test_oscura_magic_empty(self) -> None:
        """Test %oscura with no arguments shows help."""
        magics = OscuraMagics()

        with patch("builtins.print") as mock_print:
            magics.oscura("")
            mock_print.assert_called()

    def test_oscura_magic_formats(self) -> None:
        """Test %oscura formats command."""
        magics = OscuraMagics()

        with patch("oscura.get_supported_formats", return_value=["VCD", "WAV", "CSV"]):
            with patch("builtins.print") as mock_print:
                result = magics.oscura("formats")
                assert result == ["VCD", "WAV", "CSV"]
                mock_print.assert_called()

    def test_oscura_magic_unknown_command(self) -> None:
        """Test %oscura with unknown command."""
        magics = OscuraMagics()

        with patch("builtins.print") as mock_print:
            magics.oscura("unknown_command")
            mock_print.assert_called()

    def test_load_trace_success(self) -> None:
        """Test successful trace loading."""
        magics = OscuraMagics()

        # Create mock trace
        data = np.array([1.0, 2.0, 3.0])
        mock_trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e9),
        )

        with patch("oscura.load", return_value=mock_trace):
            with patch("builtins.print"):
                result = magics.oscura("load test.wfm")

        assert result == mock_trace
        assert get_current_trace() == mock_trace

        # Cleanup
        set_current_trace(None)

    def test_load_trace_error(self) -> None:
        """Test trace loading with error."""
        magics = OscuraMagics()

        with patch("oscura.load", side_effect=FileNotFoundError("Not found")):
            with patch("builtins.print") as mock_print:
                result = magics.oscura("load missing.wfm")

        assert result is None
        mock_print.assert_called()

    def test_load_trace_no_filename(self) -> None:
        """Test load command without filename."""
        magics = OscuraMagics()

        with patch("builtins.print") as mock_print:
            result = magics.oscura("load")
            assert result is None
            mock_print.assert_called()

    def test_measure_without_trace(self) -> None:
        """Test measure command without loaded trace."""
        magics = OscuraMagics()
        set_current_trace(None)

        with patch("builtins.print") as mock_print:
            result = magics.oscura("measure")
            assert result == {}
            mock_print.assert_called()

    def test_measure_all_measurements(self) -> None:
        """Test measure command with all measurements."""
        magics = OscuraMagics()

        # Create and set mock trace
        data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )
        set_current_trace(trace, "test.wfm")

        mock_results = {"rise_time": 1e-9, "fall_time": 1.2e-9}

        with patch("oscura.measure", return_value=mock_results):
            with patch.object(magics, "_display_measurements"):
                result = magics.oscura("measure")

        assert result == mock_results

        # Cleanup
        set_current_trace(None)

    def test_measure_specific_measurements(self) -> None:
        """Test measure command with specific measurements."""
        magics = OscuraMagics()

        # Create and set mock trace
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        set_current_trace(trace, "test.wfm")

        # Mock the rise_time function
        with patch("oscura.rise_time", return_value=1e-9):
            with patch.object(magics, "_display_measurements"):
                result = magics.oscura("measure rise_time")

        assert "rise_time" in result
        assert result["rise_time"] == 1e-9

        # Cleanup
        set_current_trace(None)

    def test_measure_unknown_measurement(self) -> None:
        """Test measure command with unknown measurement."""
        magics = OscuraMagics()

        # Create and set mock trace
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        set_current_trace(trace, "test.wfm")

        with patch.object(magics, "_display_measurements"):
            result = magics.oscura("measure unknown_metric")

        assert "unknown_metric" in result
        assert result["unknown_metric"] == "Unknown measurement"

        # Cleanup
        set_current_trace(None)

    def test_measure_error_in_measurement(self) -> None:
        """Test measure command when measurement raises error."""
        magics = OscuraMagics()

        # Create and set mock trace
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        set_current_trace(trace, "test.wfm")

        with patch("oscura.rise_time", side_effect=ValueError("Bad data")):
            with patch.object(magics, "_display_measurements"):
                result = magics.oscura("measure rise_time")

        assert "rise_time" in result
        assert "Error" in str(result["rise_time"])

        # Cleanup
        set_current_trace(None)

    def test_info_without_trace(self) -> None:
        """Test info command without loaded trace."""
        magics = OscuraMagics()
        set_current_trace(None)

        with patch("builtins.print") as mock_print:
            result = magics.oscura("info")
            assert result is None
            mock_print.assert_called()

    def test_info_with_trace(self) -> None:
        """Test info command with loaded trace."""
        magics = OscuraMagics()

        # Create and set mock trace
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=1e9,
                channel="CH1",
            ),
        )
        set_current_trace(trace, "test.wfm")

        with patch("builtins.print"):
            result = magics.oscura("info")

        assert result is not None
        assert result["file"] == "test.wfm"
        assert result["samples"] == 5
        assert result["sample_rate"] == 1e9
        assert result["channel"] == "CH1"

        # Cleanup
        set_current_trace(None)

    def test_display_measurements_with_ipython(self) -> None:
        """Test _display_measurements with IPython available."""
        magics = OscuraMagics()
        results = {"test": 1.0, "value": 2.5}

        with patch("oscura.jupyter.magic.IPYTHON_AVAILABLE", True):
            with patch("oscura.jupyter.display.display_measurements") as mock_display:
                magics._display_measurements(results)
                mock_display.assert_called_once_with(results)

    def test_display_measurements_without_ipython(self) -> None:
        """Test _display_measurements fallback without IPython."""
        magics = OscuraMagics()
        results = {"test": 1.0, "value": 2.5e-9}

        with patch("oscura.jupyter.magic.IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                magics._display_measurements(results)
                # Should print each measurement
                assert mock_print.call_count == 2

    def test_analyze_cell_magic(self) -> None:
        """Test %%analyze cell magic."""
        magics = OscuraMagics()

        cell_code = """
result = {"test": 42}
"""

        with patch("oscura.load"), patch("oscura.measure"):
            result = magics.analyze("", cell_code)

        assert result == {"test": 42}

    def test_analyze_cell_magic_with_trace(self) -> None:
        """Test %%analyze cell magic with current trace available."""
        magics = OscuraMagics()

        # Set current trace
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        set_current_trace(trace, "test.wfm")

        cell_code = """
result = len(trace.data)
"""

        result = magics.analyze("", cell_code)
        assert result == 3

        # Cleanup
        set_current_trace(None)

    def test_analyze_cell_magic_imports_available(self) -> None:
        """Test that analyze cell has oscura imports available."""
        magics = OscuraMagics()

        cell_code = """
# Test that imports are available in namespace
result = "load" in dir() and "measure" in dir()
"""

        result = magics.analyze("", cell_code)
        assert result is True


class TestExtensionLoading:
    """Test IPython extension loading."""

    def test_load_ipython_extension(self) -> None:
        """Test load_ipython_extension function."""
        mock_ipython = Mock()

        with patch("builtins.print"):
            load_ipython_extension(mock_ipython)

        mock_ipython.register_magics.assert_called_once()

    def test_unload_ipython_extension(self) -> None:
        """Test unload_ipython_extension function."""
        mock_ipython = Mock()

        # Should not raise
        unload_ipython_extension(mock_ipython)
