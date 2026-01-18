"""Unit tests for digital timing diagram visualization.

Tests:
"""

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.visualization.digital import plot_logic_analyzer, plot_timing

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def sample_digital_traces():
    """Create sample digital traces for testing."""
    sample_rate = 1e6  # 1 MHz
    n_samples = 1000
    metadata = TraceMetadata(sample_rate=sample_rate)

    # Clock signal
    clk_data = np.tile([False, False, True, True], n_samples // 4)[:n_samples]
    clk = DigitalTrace(data=clk_data, metadata=metadata)

    # Data signal
    data_data = np.random.randint(0, 2, n_samples).astype(bool)
    data = DigitalTrace(data=data_data, metadata=metadata)

    # Chip select (mostly low)
    cs_data = np.zeros(n_samples, dtype=bool)
    cs_data[100:500] = True
    cs = DigitalTrace(data=cs_data, metadata=metadata)

    return [clk, data, cs]


@pytest.fixture
def sample_analog_trace():
    """Create sample analog trace for conversion."""
    sample_rate = 1e6
    n_samples = 1000
    metadata = TraceMetadata(sample_rate=sample_rate)

    # Square wave
    t = np.arange(n_samples) / sample_rate
    data = 3.3 * (np.sin(2 * np.pi * 1000 * t) > 0).astype(float)

    return WaveformTrace(data=data, metadata=metadata)


class TestPlotTiming:
    """Tests for plot_timing function."""

    def test_basic_timing_plot(self, sample_digital_traces):
        """Test basic timing diagram creation."""
        pytest.importorskip("matplotlib")

        fig = plot_timing(sample_digital_traces)

        assert fig is not None
        assert len(fig.axes) == 3  # Three channels

    def test_with_names(self, sample_digital_traces):
        """Test timing plot with channel names."""
        pytest.importorskip("matplotlib")

        names = ["CLK", "DATA", "CS"]
        fig = plot_timing(sample_digital_traces, names=names)

        assert fig is not None
        # Check that labels are set
        for ax, name in zip(fig.axes, names, strict=False):
            ylabel = ax.get_ylabel()
            assert name in ylabel

    def test_time_units(self, sample_digital_traces):
        """Test different time units."""
        pytest.importorskip("matplotlib")

        for unit in ["s", "ms", "us", "ns", "auto"]:
            fig = plot_timing(sample_digital_traces, time_unit=unit)
            assert fig is not None

    def test_analog_to_digital_conversion(self, sample_analog_trace):
        """Test automatic analog-to-digital conversion."""
        pytest.importorskip("matplotlib")

        fig = plot_timing([sample_analog_trace])
        assert fig is not None

    def test_time_range(self, sample_digital_traces):
        """Test time range limiting."""
        pytest.importorskip("matplotlib")

        # Plot only middle section
        time_range = (0.0002, 0.0006)  # 200us to 600us
        fig = plot_timing(sample_digital_traces, time_range=time_range)

        assert fig is not None

    def test_empty_traces_error(self):
        """Test that empty traces list raises ValueError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(ValueError, match="traces list cannot be empty"):
            plot_timing([])

    def test_mismatched_names_error(self, sample_digital_traces):
        """Test that mismatched names length raises ValueError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(ValueError, match="names length"):
            plot_timing(sample_digital_traces, names=["CLK", "DATA"])  # Only 2 names for 3 traces

    def test_grid_toggle(self, sample_digital_traces):
        """Test grid on/off."""
        pytest.importorskip("matplotlib")

        fig_grid = plot_timing(sample_digital_traces, show_grid=True)
        fig_no_grid = plot_timing(sample_digital_traces, show_grid=False)

        assert fig_grid is not None
        assert fig_no_grid is not None

    def test_custom_title(self, sample_digital_traces):
        """Test custom title."""
        pytest.importorskip("matplotlib")

        title = "SPI Communication Timing"
        fig = plot_timing(sample_digital_traces, title=title)

        assert fig is not None
        assert fig._suptitle is not None
        assert title in fig._suptitle.get_text()


class TestPlotLogicAnalyzer:
    """Tests for plot_logic_analyzer function."""

    def test_basic_logic_analyzer(self, sample_digital_traces):
        """Test basic logic analyzer plot."""
        pytest.importorskip("matplotlib")

        fig = plot_logic_analyzer(sample_digital_traces)
        assert fig is not None

    def test_empty_traces_error(self):
        """Test that empty traces list raises ValueError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(ValueError, match="traces list cannot be empty"):
            plot_logic_analyzer([])


class TestAnnotations:
    """Tests for protocol annotation overlay."""

    def test_without_annotations(self, sample_digital_traces):
        """Test that plot works without annotations."""
        pytest.importorskip("matplotlib")

        fig = plot_timing(sample_digital_traces, annotations=None)
        assert fig is not None

    def test_with_empty_annotations(self, sample_digital_traces):
        """Test with empty annotation lists."""
        pytest.importorskip("matplotlib")

        annotations = [[], [], []]
        fig = plot_timing(sample_digital_traces, annotations=annotations)
        assert fig is not None
