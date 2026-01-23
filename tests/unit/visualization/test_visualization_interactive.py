"""Tests for interactive visualization features.

Tests requirements:
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

# Import matplotlib with backend setup for testing
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def sample_trace() -> WaveformTrace:
    """Create a sample trace for testing."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


@pytest.fixture
def bimodal_trace() -> WaveformTrace:
    """Create a trace with bimodal distribution."""
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.randn(500) - 2,
            np.random.randn(500) + 2,
        ]
    )
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


class TestPlotPhase:
    """Tests for phase plot (VIS-009)."""

    def test_plot_phase_basic(self, sample_trace: WaveformTrace) -> None:
        """Test basic phase plot creation."""
        from oscura.visualization.interactive import plot_phase

        # Create two signals with phase shift
        t = np.linspace(0, 1, 1000)
        trace1 = WaveformTrace(
            data=np.sin(2 * np.pi * 5 * t).astype(np.float64),
            metadata=TraceMetadata(sample_rate=1000.0),
        )
        trace2 = WaveformTrace(
            data=np.cos(2 * np.pi * 5 * t).astype(np.float64),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        fig, ax = plot_phase(trace1, trace2)

        assert fig is not None
        assert ax is not None
        # Phase plot should have equal aspect ratio (1.0 or "equal")
        aspect = ax.get_aspect()
        assert aspect in {"equal", 1.0}

        plt.close(fig)

    def test_plot_phase_numpy_arrays(self) -> None:
        """Test phase plot with numpy arrays."""
        from oscura.visualization.interactive import plot_phase

        t = np.linspace(0, 1, 500)
        data1 = np.sin(2 * np.pi * 3 * t)
        data2 = np.sin(2 * np.pi * 3 * t + np.pi / 4)

        fig, _ax = plot_phase(data1, data2)

        assert fig is not None
        plt.close(fig)


class TestPlotBode:
    """Tests for Bode plot (VIS-010)."""

    def test_plot_bode_mag_only(self) -> None:
        """Test Bode plot with magnitude only."""
        from oscura.visualization.interactive import plot_bode

        frequencies = np.logspace(1, 4, 100)
        magnitude = 20 * np.log10(1 / np.sqrt(1 + (frequencies / 1000) ** 2))

        fig = plot_bode(frequencies, magnitude)

        assert fig is not None
        assert len(fig.axes) >= 1  # At least magnitude axis

        plt.close(fig)

    def test_plot_bode_mag_and_phase(self) -> None:
        """Test Bode plot with magnitude and phase."""
        from oscura.visualization.interactive import plot_bode

        frequencies = np.logspace(1, 4, 100)
        magnitude = 20 * np.log10(1 / np.sqrt(1 + (frequencies / 1000) ** 2))
        phase = -np.arctan(frequencies / 1000)

        fig = plot_bode(frequencies, magnitude, phase)

        assert fig is not None
        assert len(fig.axes) >= 2  # Both magnitude and phase axes

        plt.close(fig)

    def test_plot_bode_linear_magnitude(self) -> None:
        """Test Bode plot with linear magnitude conversion."""
        from oscura.visualization.interactive import plot_bode

        frequencies = np.logspace(1, 4, 100)
        magnitude_linear = 1 / np.sqrt(1 + (frequencies / 1000) ** 2)

        fig = plot_bode(frequencies, magnitude_linear, magnitude_db=False)

        assert fig is not None
        plt.close(fig)


class TestPlotWaterfall:
    """Tests for waterfall plot (VIS-011)."""

    def test_plot_waterfall_basic(self) -> None:
        """Test basic waterfall plot creation."""
        from oscura.visualization.interactive import plot_waterfall

        # Create a chirp signal
        t = np.linspace(0, 1, 10000)
        signal = np.sin(2 * np.pi * (100 + 400 * t) * t)

        fig, ax = plot_waterfall(signal, sample_rate=10000.0, nperseg=256)

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_waterfall_custom_nperseg(self) -> None:
        """Test waterfall plot with custom segment length."""
        from oscura.visualization.interactive import plot_waterfall

        signal = np.random.randn(5000)

        fig, _ax = plot_waterfall(signal, sample_rate=1000.0, nperseg=128, noverlap=64)

        assert fig is not None
        plt.close(fig)


class TestPlotHistogram:
    """Tests for histogram plot (VIS-012)."""

    def test_plot_histogram_basic(self, sample_trace: WaveformTrace) -> None:
        """Test basic histogram plot."""
        from oscura.visualization.interactive import plot_histogram

        fig, ax, stats = plot_histogram(sample_trace, show=False)

        assert fig is not None
        assert ax is not None
        assert "mean" in stats
        assert "std" in stats
        assert "count" in stats

        plt.close(fig)

    def test_plot_histogram_with_kde(self, sample_trace: WaveformTrace) -> None:
        """Test histogram with KDE overlay."""
        from oscura.visualization.interactive import plot_histogram

        fig, ax, _stats = plot_histogram(sample_trace, show_kde=True, show=False)

        assert fig is not None
        # Check that KDE line was added (should have at least 2 lines - hist edges and KDE)
        assert len(ax.get_lines()) >= 1

        plt.close(fig)

    def test_plot_histogram_bimodal(self, bimodal_trace: WaveformTrace) -> None:
        """Test histogram on bimodal data."""
        from oscura.visualization.interactive import plot_histogram

        fig, _ax, stats = plot_histogram(bimodal_trace, bins=30, show_kde=True, show=False)

        assert fig is not None
        # Mean should be near 0 for symmetric bimodal
        assert -1 < stats["mean"] < 1

        plt.close(fig)

    def test_plot_histogram_no_stats(self, sample_trace: WaveformTrace) -> None:
        """Test histogram without statistics overlay."""
        from oscura.visualization.interactive import plot_histogram

        fig, _ax, stats = plot_histogram(sample_trace, show_stats=False, show=False)

        assert fig is not None
        assert stats is not None

        plt.close(fig)

    def test_plot_histogram_density(self, sample_trace: WaveformTrace) -> None:
        """Test histogram with density normalization."""
        from oscura.visualization.interactive import plot_histogram

        fig, ax, _stats = plot_histogram(sample_trace, density=True, show=False)

        assert fig is not None
        assert "Density" in ax.get_ylabel()

        plt.close(fig)


class TestZoomPan:
    """Tests for zoom and pan functionality (VIS-007)."""

    def test_enable_zoom_pan(self, sample_trace: WaveformTrace) -> None:
        """Test enabling zoom and pan."""
        from oscura.visualization.interactive import ZoomState, enable_zoom_pan

        fig, ax = plt.subplots()
        ax.plot(sample_trace.time_vector, sample_trace.data)

        state = enable_zoom_pan(ax)

        assert isinstance(state, ZoomState)
        assert state.home_xlim is not None
        assert state.home_ylim is not None

        plt.close(fig)

    def test_zoom_state_tracking(self, sample_trace: WaveformTrace) -> None:
        """Test that zoom state is properly tracked."""
        from oscura.visualization.interactive import enable_zoom_pan

        fig, ax = plt.subplots()
        ax.plot(sample_trace.time_vector, sample_trace.data)

        state = enable_zoom_pan(ax)

        # Initial limits should match home limits
        assert state.xlim == state.home_xlim
        assert state.ylim == state.home_ylim

        plt.close(fig)


class TestCursors:
    """Tests for measurement cursors (VIS-008)."""

    def test_plot_with_cursors(self, sample_trace: WaveformTrace) -> None:
        """Test creating plot with cursors."""
        from oscura.visualization.interactive import plot_with_cursors

        fig, ax, cursor = plot_with_cursors(sample_trace)

        assert fig is not None
        assert ax is not None
        assert cursor is not None

        plt.close(fig)

    def test_cursor_types(self, sample_trace: WaveformTrace) -> None:
        """Test different cursor types."""
        from oscura.visualization.interactive import plot_with_cursors

        for cursor_type in ["vertical", "horizontal", "cross"]:
            fig, _ax, cursor = plot_with_cursors(sample_trace, cursor_type=cursor_type)
            assert cursor is not None
            plt.close(fig)

    def test_add_measurement_cursors(self, sample_trace: WaveformTrace) -> None:
        """Test adding measurement cursors to existing plot."""
        from oscura.visualization.interactive import add_measurement_cursors

        fig, ax = plt.subplots()
        ax.plot(sample_trace.time_vector, sample_trace.data)

        cursors = add_measurement_cursors(ax)

        assert "get_measurement" in cursors
        assert callable(cursors["get_measurement"])

        # No measurement yet
        measurement = cursors["get_measurement"]()
        assert measurement is None

        plt.close(fig)

    def test_cursor_measurement_result(self) -> None:
        """Test CursorMeasurement dataclass."""
        from oscura.visualization.interactive import CursorMeasurement

        measurement = CursorMeasurement(
            x1=0.1,
            x2=0.2,
            y1=1.0,
            y2=1.5,
            delta_x=0.1,
            delta_y=0.5,
            frequency=10.0,
            slope=5.0,
        )

        assert measurement.delta_x == 0.1
        assert measurement.delta_y == 0.5
        assert measurement.frequency == 10.0
        assert measurement.slope == 5.0


class TestVisualizationInteractiveIntegration:
    """Integration tests for interactive visualization module."""

    def test_all_functions_accessible(self) -> None:
        """Test that all functions are accessible from package."""
        from oscura.visualization import (
            CursorMeasurement,
            ZoomState,
            add_measurement_cursors,
            enable_zoom_pan,
            plot_bode,
            plot_histogram,
            plot_phase,
            plot_waterfall,
            plot_with_cursors,
        )

        assert CursorMeasurement is not None
        assert ZoomState is not None
        assert callable(add_measurement_cursors)
        assert callable(enable_zoom_pan)
        assert callable(plot_bode)
        assert callable(plot_histogram)
        assert callable(plot_phase)
        assert callable(plot_waterfall)
        assert callable(plot_with_cursors)

    def test_import_from_interactive_module(self) -> None:
        """Test direct import from interactive module."""
        from oscura.visualization.interactive import (
            plot_bode,
            plot_histogram,
            plot_phase,
        )

        assert callable(plot_bode)
        assert callable(plot_histogram)
        assert callable(plot_phase)
