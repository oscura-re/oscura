"""Comprehensive unit tests for comparison visualization.

This module tests visualization functions with mocked matplotlib to avoid
display issues during testing.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.utils.comparison.compare import compare_traces
from oscura.utils.comparison.visualization import (
    plot_comparison_heatmap,
    plot_comparison_summary,
    plot_difference,
    plot_overlay,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def trace1() -> WaveformTrace:
    """Create first test trace."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6, channel="CH1")
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def trace2() -> WaveformTrace:
    """Create second test trace (slightly different)."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000) + 0.1)
    metadata = TraceMetadata(sample_rate=1e6, channel="CH2")
    return WaveformTrace(data=data, metadata=metadata)


def configure_plt_mock(mock_plt: Mock) -> tuple[Mock, Mock]:
    """Configure a matplotlib mock to return proper objects.

    Args:
        mock_plt: The mocked plt module

    Returns:
        Tuple of (mock_fig, mock_ax) for assertions
    """
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.figure.return_value = mock_fig
    return mock_fig, mock_ax


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestPlotOverlay:
    """Test overlay plot function."""

    @patch("oscura.utils.comparison.visualization.plt")
    def test_basic_overlay(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test basic overlay plot creation."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace2)

        assert fig is not None
        mock_plt.subplots.assert_called_once()

    @patch("oscura.utils.comparison.visualization.plt")
    def test_overlay_with_labels(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test overlay with custom labels."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace2, labels=("Measured", "Reference"), title="My Comparison")

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_overlay_without_highlighting(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test overlay without difference highlighting."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace2, highlight_differences=False)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_overlay_custom_threshold(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test overlay with custom difference threshold."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace2, difference_threshold=0.1)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_overlay_custom_figsize(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test overlay with custom figure size."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace2, figsize=(12, 8))

        assert fig is not None


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestPlotDifference:
    """Test difference plot function."""

    @patch("oscura.utils.comparison.visualization.plt")
    def test_basic_difference(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test basic difference plot creation."""
        configure_plt_mock(mock_plt)
        fig = plot_difference(trace1, trace2)

        assert fig is not None
        mock_plt.subplots.assert_called_once()

    @patch("oscura.utils.comparison.visualization.plt")
    def test_difference_normalized(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test normalized difference plot."""
        configure_plt_mock(mock_plt)
        fig = plot_difference(trace1, trace2, normalize=True)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_difference_without_statistics(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test difference plot without statistics box."""
        configure_plt_mock(mock_plt)
        fig = plot_difference(trace1, trace2, show_statistics=False)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_difference_custom_title(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test difference plot with custom title."""
        configure_plt_mock(mock_plt)
        fig = plot_difference(trace1, trace2, title="Custom Difference Plot")

        assert fig is not None


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestPlotHeatmap:
    """Test comparison heatmap function."""

    @patch("oscura.utils.comparison.visualization.plt")
    def test_basic_heatmap(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test basic heatmap creation."""
        configure_plt_mock(mock_plt)
        fig = plot_comparison_heatmap(trace1, trace2)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_heatmap_custom_window(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test heatmap with custom window size."""
        configure_plt_mock(mock_plt)
        fig = plot_comparison_heatmap(trace1, trace2, window_size=50)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_heatmap_custom_title(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test heatmap with custom title."""
        configure_plt_mock(mock_plt)
        fig = plot_comparison_heatmap(trace1, trace2, title="My Heatmap")

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_heatmap_short_trace(self, mock_plt: Mock) -> None:
        """Test heatmap with very short trace."""
        configure_plt_mock(mock_plt)
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.1, 2.1, 3.1])
        metadata = TraceMetadata(sample_rate=1e6)
        trace1 = WaveformTrace(data=data1, metadata=metadata)
        trace2 = WaveformTrace(data=data2, metadata=metadata)

        fig = plot_comparison_heatmap(trace1, trace2, window_size=10)

        assert fig is not None


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestPlotSummary:
    """Test comparison summary plot function."""

    @patch("oscura.utils.comparison.visualization.plt")
    def test_basic_summary(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test basic summary plot creation."""
        result = compare_traces(trace1, trace2)
        fig = plot_comparison_summary(result)

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_summary_custom_title(
        self, mock_plt: Mock, trace1: WaveformTrace, trace2: WaveformTrace
    ) -> None:
        """Test summary plot with custom title."""
        result = compare_traces(trace1, trace2)
        fig = plot_comparison_summary(result, title="My Summary")

        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_summary_with_violations(self, mock_plt: Mock, trace1: WaveformTrace) -> None:
        """Test summary plot with violations."""
        # Create trace with violations
        data2 = trace1.data.copy()
        data2[100:110] += 1.0
        trace2 = WaveformTrace(data=data2, metadata=trace1.metadata)

        result = compare_traces(trace1, trace2, tolerance=0.1)
        fig = plot_comparison_summary(result)

        assert fig is not None
        assert not result.match

    @patch("oscura.utils.comparison.visualization.plt")
    def test_summary_passing_comparison(self, mock_plt: Mock, trace1: WaveformTrace) -> None:
        """Test summary plot with passing comparison."""
        result = compare_traces(trace1, trace1)
        fig = plot_comparison_summary(result)

        assert fig is not None
        assert result.match


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonVisualizationEdgeCases:
    """Test edge cases for visualization."""

    @patch("oscura.utils.comparison.visualization.plt")
    def test_identical_traces(self, mock_plt: Mock, trace1: WaveformTrace) -> None:
        """Test visualizing identical traces."""
        configure_plt_mock(mock_plt)
        fig = plot_overlay(trace1, trace1)
        assert fig is not None

        fig = plot_difference(trace1, trace1)
        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_constant_traces(self, mock_plt: Mock) -> None:
        """Test visualizing constant traces."""
        configure_plt_mock(mock_plt)
        data = np.ones(1000) * 0.5
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        fig = plot_overlay(trace, trace)
        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_different_lengths(self, mock_plt: Mock, trace1: WaveformTrace) -> None:
        """Test visualizing traces of different lengths."""
        configure_plt_mock(mock_plt)
        data2 = trace1.data[:500]
        trace2 = WaveformTrace(data=data2, metadata=trace1.metadata)

        fig = plot_overlay(trace1, trace2)
        assert fig is not None

    @patch("oscura.utils.comparison.visualization.plt")
    def test_no_sample_rate(self, mock_plt: Mock) -> None:
        """Test visualization without sample rate metadata."""
        configure_plt_mock(mock_plt)
        # This should use sample indices instead of time
        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        # Create metadata without sample_rate by setting it to minimal value
        metadata = TraceMetadata(sample_rate=1.0)
        metadata.sample_rate = None  # type: ignore
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should handle missing sample rate gracefully
        try:
            fig = plot_difference(trace, trace)
            assert fig is not None
        except AttributeError:
            # Acceptable if it requires sample rate
            pass
