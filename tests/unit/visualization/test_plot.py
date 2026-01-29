"""Comprehensive tests for visualization.plot module.

Tests cover the plot namespace module that provides convenient imports
and wrappers for plotting functions.

Coverage target: >90%
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.visualization import plot

pytestmark = [pytest.mark.usefixtures("cleanup_matplotlib")]


class TestPlotTrace:
    """Tests for plot_trace function."""

    def test_plot_trace_calls_plot_waveform(self) -> None:
        """Test that plot_trace delegates to plot_waveform."""
        # Create mock trace
        trace = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        with patch("oscura.visualization.plot.plot_waveform") as mock_plot:
            plot.plot_trace(trace, title="Test")

            # Should call plot_waveform with same arguments
            mock_plot.assert_called_once()
            args, kwargs = mock_plot.call_args
            assert args[0] is trace
            assert kwargs["title"] == "Test"

    def test_plot_trace_with_kwargs(self) -> None:
        """Test that plot_trace passes through kwargs."""
        trace = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        with patch("oscura.visualization.plot.plot_waveform") as mock_plot:
            plot.plot_trace(trace, color="red", linewidth=2)

            # Should pass kwargs
            args, kwargs = mock_plot.call_args
            assert kwargs["color"] == "red"
            assert kwargs["linewidth"] == 2


class TestAddAnnotation:
    """Tests for add_annotation function."""

    @pytest.fixture
    def matplotlib_available(self) -> None:
        """Ensure matplotlib is available for these tests."""
        pytest.importorskip("matplotlib")

    def test_add_annotation_basic(self, matplotlib_available: None) -> None:
        """Test basic annotation functionality."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Add annotation
        plot.add_annotation("Test annotation")

        # Check that annotation was added
        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_annotation_with_kwargs(self, matplotlib_available: None) -> None:
        """Test annotation with kwargs."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Add annotation with custom properties
        plot.add_annotation("Test", fontsize=14, color="red")

        # Annotation should be present
        assert len(ax.texts) > 0
        # Note: Can't easily verify properties were passed through
        # without inspecting text object internals

        plt.close(fig)


class TestModuleStructure:
    """Tests for module structure and exports."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected functions."""
        expected_exports = [
            "add_annotation",
            "plot_bathtub",
            "plot_bode",
            "plot_eye",
            "plot_fft",
            "plot_histogram",
            "plot_logic_analyzer",
            "plot_multi_channel",
            "plot_phase",
            "plot_psd",
            "plot_spectrogram",
            "plot_spectrum",
            "plot_timing",
            "plot_trace",
            "plot_waterfall",
            "plot_waveform",
            "plot_xy",
        ]

        for export in expected_exports:
            assert export in plot.__all__

    def test_function_imports(self) -> None:
        """Test that functions are properly imported."""
        # Digital functions
        assert hasattr(plot, "plot_logic_analyzer")
        assert hasattr(plot, "plot_timing")

        # Eye diagram functions
        assert hasattr(plot, "plot_eye")
        assert hasattr(plot, "plot_bathtub")

        # Interactive functions
        assert hasattr(plot, "plot_bode")
        assert hasattr(plot, "plot_histogram")
        assert hasattr(plot, "plot_phase")
        assert hasattr(plot, "plot_waterfall")

        # Spectral functions
        assert hasattr(plot, "plot_fft")
        assert hasattr(plot, "plot_psd")
        assert hasattr(plot, "plot_spectrogram")
        assert hasattr(plot, "plot_spectrum")

        # Waveform functions
        assert hasattr(plot, "plot_multi_channel")
        assert hasattr(plot, "plot_waveform")
        assert hasattr(plot, "plot_xy")

        # Local functions
        assert hasattr(plot, "plot_trace")
        assert hasattr(plot, "add_annotation")

    def test_imported_functions_are_callable(self) -> None:
        """Test that imported functions are callable."""
        assert callable(plot.plot_logic_analyzer)
        assert callable(plot.plot_timing)
        assert callable(plot.plot_eye)
        assert callable(plot.plot_bathtub)
        assert callable(plot.plot_bode)
        assert callable(plot.plot_histogram)
        assert callable(plot.plot_phase)
        assert callable(plot.plot_waterfall)
        assert callable(plot.plot_fft)
        assert callable(plot.plot_psd)
        assert callable(plot.plot_spectrogram)
        assert callable(plot.plot_spectrum)
        assert callable(plot.plot_multi_channel)
        assert callable(plot.plot_waveform)
        assert callable(plot.plot_xy)
        assert callable(plot.plot_trace)
        assert callable(plot.add_annotation)


class TestIntegration:
    """Integration tests for plot module usage."""

    def test_plot_namespace_usage(self) -> None:
        """Test using plot namespace as intended."""
        # This is the intended usage pattern:
        # from oscura.visualization import plot
        # plot.plot_trace(trace)

        # Verify namespace works
        assert hasattr(plot, "plot_trace")
        assert hasattr(plot, "plot_waveform")

    def test_function_delegation_chain(self) -> None:
        """Test that plot_trace delegates through to actual plotting."""
        trace = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        # Mock the underlying plot_waveform at module import level
        with patch("oscura.visualization.plot.plot_waveform") as mock:
            # Call plot_trace which should delegate to plot_waveform
            plot.plot_trace(trace)

            # Should have called the mocked function
            assert mock.called


# Run tests with: pytest tests/unit/visualization/test_plot.py -v
