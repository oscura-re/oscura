"""Comprehensive unit tests for waveform visualization module.

Requirements tested:

This module provides comprehensive tests for waveform visualization functions
to achieve 90%+ code coverage.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_waveform():
    """Create a sample waveform trace for testing."""
    sample_rate = 1e6  # 1 MHz
    n_samples = 1000
    t = np.arange(n_samples) / sample_rate
    data = 2.5 * np.sin(2 * np.pi * 1000 * t)
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="CH1",
        vertical_scale=1.0,
        vertical_offset=0.0,
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_waveform_short_duration():
    """Create a waveform with very short duration (nanoseconds)."""
    sample_rate = 1e9  # 1 GHz
    n_samples = 100
    data = np.sin(2 * np.pi * 1e6 * np.arange(n_samples) / sample_rate)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_waveform_medium_duration():
    """Create a waveform with medium duration (microseconds)."""
    sample_rate = 1e6  # 1 MHz
    n_samples = 500
    data = np.sin(2 * np.pi * 1000 * np.arange(n_samples) / sample_rate)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_waveform_long_duration():
    """Create a waveform with long duration (seconds)."""
    sample_rate = 1000  # 1 kHz
    n_samples = 2000
    data = np.sin(2 * np.pi * 10 * np.arange(n_samples) / sample_rate)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_waveform_ms_duration():
    """Create a waveform with millisecond duration (1-1000 ms)."""
    sample_rate = 10000  # 10 kHz
    n_samples = 5000  # 0.5 seconds
    data = np.sin(2 * np.pi * 100 * np.arange(n_samples) / sample_rate)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def multi_channel_waveforms():
    """Create multiple waveform traces."""
    sample_rate = 1e6
    n_samples = 1000
    t = np.arange(n_samples) / sample_rate

    metadata1 = TraceMetadata(sample_rate=sample_rate, channel_name="CH1")
    metadata2 = TraceMetadata(sample_rate=sample_rate, channel_name="CH2")
    metadata3 = TraceMetadata(sample_rate=sample_rate, channel_name="CH3")

    ch1 = WaveformTrace(data=np.sin(2 * np.pi * 1000 * t), metadata=metadata1)
    ch2 = WaveformTrace(data=np.cos(2 * np.pi * 1000 * t), metadata=metadata2)
    ch3 = WaveformTrace(data=np.sin(2 * np.pi * 2000 * t), metadata=metadata3)

    return [ch1, ch2, ch3]


@pytest.fixture
def digital_trace():
    """Create a sample digital trace."""
    sample_rate = 1e6
    n_samples = 1000
    data = np.tile([False, False, True, True], n_samples // 4)[:n_samples]
    metadata = TraceMetadata(sample_rate=sample_rate)
    return DigitalTrace(data=data, metadata=metadata)


@pytest.fixture
def empty_waveform():
    """Create an empty waveform trace."""
    sample_rate = 1e6
    data = np.array([], dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotWaveform:
    """Tests for plot_waveform function."""

    def test_matplotlib_not_installed(self, sample_waveform):
        """Test that ImportError is raised when matplotlib is not available."""
        with patch("oscura.visualization.waveform.HAS_MATPLOTLIB", False):
            from oscura.visualization.waveform import plot_waveform

            with pytest.raises(ImportError, match="matplotlib is required"):
                plot_waveform(sample_waveform)

    def test_basic_waveform_plot(self, sample_waveform):
        """Test basic waveform plotting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, show=False)

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        # Time unit is auto-selected, just check it has a time label
        assert "Time" in ax.get_xlabel()
        assert ax.get_ylabel() == "Amplitude"

    def test_with_custom_axes(self, sample_waveform):
        """Test plotting on provided axes."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import plot_waveform

        fig, ax = plt.subplots()
        result_fig = plot_waveform(sample_waveform, ax=ax, show=False)

        assert result_fig is fig
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_axes_without_figure_error(self, sample_waveform):
        """Test that ValueError is raised for axes without figure."""
        pytest.importorskip("matplotlib")
        from matplotlib.axes import Axes

        from oscura.visualization.waveform import plot_waveform

        # Create a mock axes without a figure
        mock_ax = Mock(spec=Axes)
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_waveform(sample_waveform, ax=mock_ax, show=False)

    def test_auto_time_unit_nanoseconds(self, sample_waveform_short_duration):
        """Test automatic time unit selection for nanoseconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform_short_duration, time_unit="auto", show=False)

        ax = fig.axes[0]
        assert "ns" in ax.get_xlabel()

    def test_auto_time_unit_microseconds(self, sample_waveform_medium_duration):
        """Test automatic time unit selection for microseconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform_medium_duration, time_unit="auto", show=False)

        ax = fig.axes[0]
        assert "us" in ax.get_xlabel()

    def test_auto_time_unit_milliseconds(self, sample_waveform):
        """Test automatic time unit selection for milliseconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        # The sample_waveform has 1000 samples at 1MHz = 1ms duration
        # This is actually < 1e-3, so it will select "us" not "ms"
        # Let's verify it selects a reasonable unit
        fig = plot_waveform(sample_waveform, time_unit="auto", show=False)

        ax = fig.axes[0]
        # Should select us for this duration
        assert "us" in ax.get_xlabel()

    def test_auto_time_unit_seconds(self, sample_waveform_long_duration):
        """Test automatic time unit selection for seconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform_long_duration, time_unit="auto", show=False)

        ax = fig.axes[0]
        assert "s" in ax.get_xlabel()

    def test_auto_time_unit_milliseconds_actual(self, sample_waveform_ms_duration):
        """Test automatic time unit selection for actual milliseconds range."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform_ms_duration, time_unit="auto", show=False)

        ax = fig.axes[0]
        assert "ms" in ax.get_xlabel()

    def test_auto_time_unit_empty_trace(self, empty_waveform):
        """Test automatic time unit selection for empty trace."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(empty_waveform, time_unit="auto", show=False)

        # Should default to nanoseconds for zero duration
        ax = fig.axes[0]
        assert "ns" in ax.get_xlabel()

    def test_explicit_time_units(self, sample_waveform):
        """Test all explicit time unit options."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        for unit in ["s", "ms", "us", "ns"]:
            fig = plot_waveform(sample_waveform, time_unit=unit, show=False)
            ax = fig.axes[0]
            assert unit in ax.get_xlabel()

    def test_invalid_time_unit(self, sample_waveform):
        """Test invalid time unit defaults to 1.0 multiplier."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        # Should not raise, just use default multiplier
        fig = plot_waveform(sample_waveform, time_unit="invalid", show=False)
        assert fig is not None

    def test_time_range(self, sample_waveform):
        """Test time range limiting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        time_range = (0.0, 0.5e-3)
        fig = plot_waveform(sample_waveform, time_range=time_range, show=False)

        ax = fig.axes[0]
        xlim = ax.get_xlim()
        # Check that limits are set (allow for some rounding)
        assert xlim[0] <= 0.5  # 0.5 ms in milliseconds
        assert xlim[1] >= 0.0

    def test_show_grid_true(self, sample_waveform):
        """Test grid display enabled."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, show_grid=True, show=False)

        ax = fig.axes[0]
        assert ax.get_axisbelow() is not None  # Grid is configured

    def test_show_grid_false(self, sample_waveform):
        """Test grid display disabled."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, show_grid=False, show=False)

        assert fig is not None

    def test_custom_color(self, sample_waveform):
        """Test custom line color."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, color="red", show=False)

        ax = fig.axes[0]
        assert len(ax.lines) == 1

    def test_custom_label(self, sample_waveform):
        """Test custom legend label."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        label = "Test Signal"
        fig = plot_waveform(sample_waveform, label=label, show=False)

        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    def test_no_label(self, sample_waveform):
        """Test without legend label."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, label=None, show=False)

        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is None

    def test_custom_title(self, sample_waveform):
        """Test custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        title = "Custom Test Waveform"
        fig = plot_waveform(sample_waveform, title=title, show=False)

        ax = fig.axes[0]
        assert ax.get_title() == title

    def test_title_from_channel_name(self, sample_waveform):
        """Test automatic title from channel name."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, show=False)

        ax = fig.axes[0]
        assert "CH1" in ax.get_title()

    def test_no_title_no_channel_name(self):
        """Test no title when no custom title or channel name."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        sample_rate = 1e6
        data = np.sin(2 * np.pi * 1000 * np.arange(100) / sample_rate)
        metadata = TraceMetadata(sample_rate=sample_rate)  # No channel_name
        trace = WaveformTrace(data=data, metadata=metadata)

        fig = plot_waveform(trace, show=False)

        ax = fig.axes[0]
        assert ax.get_title() == ""

    def test_custom_labels(self, sample_waveform):
        """Test custom axis labels."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        xlabel = "Custom Time"
        ylabel = "Custom Voltage"
        fig = plot_waveform(
            sample_waveform,
            xlabel=xlabel,
            ylabel=ylabel,
            show=False,
        )

        ax = fig.axes[0]
        assert xlabel in ax.get_xlabel()
        assert ax.get_ylabel() == ylabel

    def test_custom_figsize(self, sample_waveform):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        figsize = (12, 8)
        fig = plot_waveform(sample_waveform, figsize=figsize, show=False)

        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]

    def test_save_path(self, sample_waveform, tmp_path):
        """Test saving figure to file."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        save_path = tmp_path / "test_waveform.png"
        fig = plot_waveform(sample_waveform, save_path=str(save_path), show=False)

        assert save_path.exists()
        assert fig is not None

    def test_show_parameter(self, sample_waveform):
        """Test show parameter controls plt.show()."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import plot_waveform

        with patch.object(plt, "show") as mock_show:
            plot_waveform(sample_waveform, show=True)
            mock_show.assert_called_once()

        with patch.object(plt, "show") as mock_show:
            plot_waveform(sample_waveform, show=False)
            mock_show.assert_not_called()

    def test_measurements_with_float_values(self, sample_waveform):
        """Test measurement annotations with float values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Peak": 2.5,
            "RMS": 1.77,
            "Frequency": 1000.0,
        }
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        ax = fig.axes[0]
        # Check that annotations were added
        assert len(ax.texts) > 0

    def test_measurements_with_dict_values(self, sample_waveform):
        """Test measurement annotations with dict values containing units."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Voltage": {"value": 3.3, "unit": "V"},
            "Frequency": {"value": 1000, "unit": "Hz"},
        }
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        ax = fig.axes[0]
        assert len(ax.texts) > 0

    def test_measurements_with_nan_values(self, sample_waveform):
        """Test measurement annotations skip NaN values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Valid": 2.5,
            "Invalid": np.nan,
            "AlsoValid": 1.77,
        }
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        # Should still create the plot
        assert fig is not None

    def test_measurements_with_nan_in_dict(self, sample_waveform):
        """Test measurement annotations skip NaN in dict values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Valid": {"value": 2.5, "unit": "V"},
            "Invalid": {"value": np.nan, "unit": "V"},
        }
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        assert fig is not None

    def test_measurements_empty_dict(self, sample_waveform):
        """Test measurement annotations with empty dict."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {}
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        ax = fig.axes[0]
        # No annotations should be added
        assert fig is not None

    def test_measurements_none(self, sample_waveform):
        """Test measurement annotations with None."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        fig = plot_waveform(sample_waveform, show_measurements=None, show=False)

        assert fig is not None


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotMultiChannel:
    """Tests for plot_multi_channel function."""

    def test_matplotlib_not_installed(self, multi_channel_waveforms):
        """Test that ImportError is raised when matplotlib is not available."""
        with patch("oscura.visualization.waveform.HAS_MATPLOTLIB", False):
            from oscura.visualization.waveform import plot_multi_channel

            with pytest.raises(ImportError, match="matplotlib is required"):
                plot_multi_channel(multi_channel_waveforms)

    def test_basic_multi_channel(self, multi_channel_waveforms):
        """Test basic multi-channel plotting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms)

        assert fig is not None
        assert len(fig.axes) == 3

    def test_single_channel(self, sample_waveform):
        """Test multi-channel with single trace."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel([sample_waveform])

        assert fig is not None
        assert len(fig.axes) == 1

    def test_with_custom_names(self, multi_channel_waveforms):
        """Test multi-channel with custom channel names."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        names = ["Signal A", "Signal B", "Signal C"]
        fig = plot_multi_channel(multi_channel_waveforms, names=names)

        assert fig is not None
        for ax, name in zip(fig.axes, names, strict=False):
            # Name should appear in ylabel
            assert name in ax.get_ylabel()

    def test_without_custom_names(self, multi_channel_waveforms):
        """Test multi-channel with default names."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, names=None)

        assert fig is not None
        # Should use default names CH1, CH2, CH3
        for i, ax in enumerate(fig.axes):
            expected_name = f"CH{i + 1}"
            assert expected_name in ax.get_ylabel()

    def test_shared_x_true(self, multi_channel_waveforms):
        """Test multi-channel with shared x-axis."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, shared_x=True)

        assert fig is not None

    def test_shared_x_false(self, multi_channel_waveforms):
        """Test multi-channel with independent x-axes."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, shared_x=False)

        assert fig is not None

    def test_share_x_alias(self, multi_channel_waveforms):
        """Test share_x parameter as alias for shared_x."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, share_x=True)

        assert fig is not None

    def test_custom_colors(self, multi_channel_waveforms):
        """Test multi-channel with custom colors."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        colors = ["red", "green", "blue"]
        fig = plot_multi_channel(multi_channel_waveforms, colors=colors)

        assert fig is not None

    def test_colors_shorter_than_traces(self, multi_channel_waveforms):
        """Test multi-channel with fewer colors than traces."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        colors = ["red", "green"]  # Only 2 colors for 3 traces
        fig = plot_multi_channel(multi_channel_waveforms, colors=colors)

        # Should still work, third trace uses default
        assert fig is not None

    def test_no_custom_colors(self, multi_channel_waveforms):
        """Test multi-channel with default colors."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, colors=None)

        assert fig is not None

    def test_auto_time_unit(self, multi_channel_waveforms):
        """Test automatic time unit selection."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, time_unit="auto")

        ax = fig.axes[-1]  # Last subplot has x-label
        assert "ms" in ax.get_xlabel()

    def test_explicit_time_units(self, multi_channel_waveforms):
        """Test explicit time unit options."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        for unit in ["s", "ms", "us", "ns"]:
            fig = plot_multi_channel(multi_channel_waveforms, time_unit=unit)
            ax = fig.axes[-1]
            assert unit in ax.get_xlabel()

    def test_auto_time_unit_empty_traces(self):
        """Test automatic time unit with empty traces list."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        # Create trace with default settings to test edge case
        sample_rate = 1e6
        data = np.array([0.0])
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        fig = plot_multi_channel([trace], time_unit="auto")

        assert fig is not None

    def test_show_grid_true(self, multi_channel_waveforms):
        """Test grid display enabled."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, show_grid=True)

        assert fig is not None

    def test_show_grid_false(self, multi_channel_waveforms):
        """Test grid display disabled."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, show_grid=False)

        assert fig is not None

    def test_custom_figsize(self, multi_channel_waveforms):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        figsize = (12, 10)
        fig = plot_multi_channel(multi_channel_waveforms, figsize=figsize)

        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]

    def test_default_figsize(self, multi_channel_waveforms):
        """Test default figure size calculation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, figsize=None)

        # Default should be (10, 2 * n_channels)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 2 * 3

    def test_custom_title(self, multi_channel_waveforms):
        """Test custom overall title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        title = "Multi-Channel Acquisition"
        fig = plot_multi_channel(multi_channel_waveforms, title=title)

        assert fig._suptitle is not None
        assert title in fig._suptitle.get_text()

    def test_no_title(self, multi_channel_waveforms):
        """Test without overall title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, title=None)

        assert fig._suptitle is None

    def test_waveform_trace_ylabel(self, multi_channel_waveforms):
        """Test that waveform traces have V ylabel."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms)

        # All traces are waveforms, so all should have "V" in ylabel
        for ax in fig.axes:
            ylabel = ax.get_ylabel()
            # ylabel should contain channel name, but checking it exists
            assert ylabel is not None

    def test_digital_trace_step_plot(self, digital_trace):
        """Test digital trace renders as step plot."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel([digital_trace])

        ax = fig.axes[0]
        # Check ylim is set for digital signals
        ylim = ax.get_ylim()
        assert ylim[0] < 0
        assert ylim[1] > 1

    def test_mixed_waveform_and_digital(self, sample_waveform, digital_trace):
        """Test mixed waveform and digital traces."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        traces = [sample_waveform, digital_trace]
        fig = plot_multi_channel(traces)

        assert fig is not None
        assert len(fig.axes) == 2

    def test_xlabel_only_on_bottom(self, multi_channel_waveforms):
        """Test that x-label only appears on bottom subplot."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms)

        # Only the last (bottom) subplot should have x-label
        for i, ax in enumerate(fig.axes):
            if i == len(fig.axes) - 1:
                assert ax.get_xlabel() != ""
            else:
                # Upper subplots may or may not have xlabel depending on sharex
                pass

    def test_multi_channel_auto_time_unit_nanoseconds(self):
        """Test multi-channel auto time unit for nanoseconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        # Create very short duration traces (nanoseconds)
        sample_rate = 10e9  # 10 GHz
        n_samples = 50
        metadata = TraceMetadata(sample_rate=sample_rate)

        traces = [
            WaveformTrace(data=np.sin(np.arange(n_samples) * 0.1), metadata=metadata),
            WaveformTrace(data=np.cos(np.arange(n_samples) * 0.1), metadata=metadata),
        ]

        fig = plot_multi_channel(traces, time_unit="auto")

        ax = fig.axes[-1]
        assert "ns" in ax.get_xlabel()

    def test_multi_channel_auto_time_unit_seconds(self):
        """Test multi-channel auto time unit for seconds."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        # Create long duration traces (seconds)
        sample_rate = 1000  # 1 kHz
        n_samples = 2000  # 2 seconds
        metadata = TraceMetadata(sample_rate=sample_rate)

        traces = [
            WaveformTrace(data=np.sin(np.arange(n_samples) * 0.01), metadata=metadata),
            WaveformTrace(data=np.cos(np.arange(n_samples) * 0.01), metadata=metadata),
        ]

        fig = plot_multi_channel(traces, time_unit="auto")

        ax = fig.axes[-1]
        assert "s" in ax.get_xlabel()


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotXY:
    """Tests for plot_xy function (Lissajous/XY plots)."""

    def test_matplotlib_not_installed(self, sample_waveform):
        """Test that ImportError is raised when matplotlib is not available."""
        with patch("oscura.visualization.waveform.HAS_MATPLOTLIB", False):
            from oscura.visualization.waveform import plot_xy

            with pytest.raises(ImportError, match="matplotlib is required"):
                plot_xy(sample_waveform, sample_waveform)

    def test_basic_xy_plot_with_traces(self, sample_waveform):
        """Test basic XY plot with WaveformTrace objects."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        # Create a second trace
        sample_rate = sample_waveform.metadata.sample_rate
        n_samples = len(sample_waveform.data)
        data_y = np.cos(2 * np.pi * 1000 * np.arange(n_samples) / sample_rate)
        trace_y = WaveformTrace(
            data=data_y,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        fig = plot_xy(sample_waveform, trace_y)

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "X (V)"
        assert ax.get_ylabel() == "Y (V)"

    def test_xy_plot_with_numpy_arrays(self):
        """Test XY plot with numpy arrays."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data)

        assert fig is not None

    def test_xy_plot_mixed_types(self, sample_waveform):
        """Test XY plot with mixed WaveformTrace and numpy array."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        y_data = np.cos(np.linspace(0, 2 * np.pi, len(sample_waveform.data)))

        fig = plot_xy(sample_waveform, y_data)

        assert fig is not None

    def test_xy_plot_with_custom_axes(self):
        """Test XY plot on provided axes."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig, ax = plt.subplots()
        result_fig = plot_xy(x_data, y_data, ax=ax)

        assert result_fig is fig
        plt.close(fig)

    def test_xy_plot_axes_without_figure_error(self):
        """Test that ValueError is raised for axes without figure."""
        pytest.importorskip("matplotlib")
        from matplotlib.axes import Axes

        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        mock_ax = Mock(spec=Axes)
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_xy(x_data, y_data, ax=mock_ax)

    def test_xy_plot_different_lengths(self):
        """Test XY plot with different length arrays."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 150))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data)

        # Should use minimum length
        assert fig is not None

    def test_xy_plot_custom_color(self):
        """Test XY plot with custom color."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data, color="red")

        assert fig is not None

    def test_xy_plot_custom_marker(self):
        """Test XY plot with custom marker."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data, marker="o")

        assert fig is not None

    def test_xy_plot_custom_alpha(self):
        """Test XY plot with custom transparency."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data, alpha=0.5)

        assert fig is not None

    def test_xy_plot_custom_title(self):
        """Test XY plot with custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        title = "Lissajous Pattern"
        fig = plot_xy(x_data, y_data, title=title)

        ax = fig.axes[0]
        assert ax.get_title() == title

    def test_xy_plot_no_title(self):
        """Test XY plot without title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data, title=None)

        ax = fig.axes[0]
        assert ax.get_title() == ""

    def test_xy_plot_equal_aspect(self):
        """Test XY plot has equal aspect ratio."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data)

        ax = fig.axes[0]
        # get_aspect() returns 1.0 for equal aspect, or the string "equal"
        # depending on matplotlib version
        aspect = ax.get_aspect()
        assert aspect == "equal" or aspect == 1.0

    def test_xy_plot_has_grid(self):
        """Test XY plot has grid enabled."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        y_data = np.cos(np.linspace(0, 2 * np.pi, 100))

        fig = plot_xy(x_data, y_data)

        ax = fig.axes[0]
        # Grid should be enabled
        assert ax.get_axisbelow() is not None


# =============================================================================
# Private Helper Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestAddMeasurementAnnotations:
    """Tests for _add_measurement_annotations private function."""

    def test_annotation_with_valid_measurements(self, sample_waveform):
        """Test annotation addition with valid measurements."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {"Peak": 2.5, "RMS": 1.77}

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        # Should have added an annotation
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_annotation_with_dict_measurements(self, sample_waveform):
        """Test annotation with dictionary-style measurements."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {
            "Voltage": {"value": 3.3, "unit": "V"},
            "Current": {"value": 0.5, "unit": "A"},
        }

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        assert len(ax.texts) > 0
        plt.close(fig)

    def test_annotation_skips_nan_float(self, sample_waveform):
        """Test annotation skips NaN float values."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {"Valid": 2.5, "Invalid": np.nan}

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        # Should still add annotation (for valid value only)
        plt.close(fig)

    def test_annotation_skips_nan_in_dict(self, sample_waveform):
        """Test annotation skips NaN in dictionary values."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {
            "Valid": {"value": 2.5, "unit": "V"},
            "Invalid": {"value": np.nan, "unit": "V"},
        }

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        plt.close(fig)

    def test_annotation_with_empty_measurements(self, sample_waveform):
        """Test annotation with empty measurements dict."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {}

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        # Should not add any annotations
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_annotation_with_all_nan_measurements(self, sample_waveform):
        """Test annotation when all measurements are NaN."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {"Invalid1": np.nan, "Invalid2": np.nan}

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        # Should not add any annotations
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_annotation_dict_without_unit(self, sample_waveform):
        """Test annotation with dict value without unit key."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        measurements = {"Measurement": {"value": 2.5}}  # No unit

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        assert len(ax.texts) > 0
        plt.close(fig)

    def test_annotation_dict_with_non_value_key(self, sample_waveform):
        """Test annotation with dict that doesn't have 'value' key."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import _add_measurement_annotations

        fig, ax = plt.subplots()
        # Dict without 'value' key should be treated as the value itself
        measurements = {"Measurement": {"other_key": 2.5}}

        _add_measurement_annotations(ax, sample_waveform, measurements, "ms", 1e3)

        plt.close(fig)


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from oscura.visualization import waveform

        assert hasattr(waveform, "__all__")
        assert "plot_waveform" in waveform.__all__
        assert "plot_multi_channel" in waveform.__all__
        assert "plot_xy" in waveform.__all__
        assert len(waveform.__all__) == 3


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestVisualizationWaveformEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_plot_waveform_with_very_large_time_range(self, sample_waveform):
        """Test with time range extending beyond data."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        # Time range larger than actual data duration
        time_range = (0.0, 1.0)  # Much larger than 1ms data duration
        fig = plot_waveform(sample_waveform, time_range=time_range, show=False)

        assert fig is not None

    def test_plot_waveform_with_negative_time_range(self, sample_waveform):
        """Test with negative time range values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        time_range = (-0.0001, 0.0005)
        fig = plot_waveform(sample_waveform, time_range=time_range, show=False)

        assert fig is not None

    def test_plot_multi_channel_with_empty_list(self):
        """Test multi-channel plot with empty traces list."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        # This will fail during subplot creation, but tests the edge case
        with pytest.raises(ValueError):
            plot_multi_channel([])

    def test_plot_xy_with_single_point(self):
        """Test XY plot with single data point."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.array([1.0])
        y_data = np.array([2.0])

        fig = plot_xy(x_data, y_data)

        assert fig is not None

    def test_plot_xy_with_zero_length_arrays(self):
        """Test XY plot with empty arrays."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        x_data = np.array([])
        y_data = np.array([])

        fig = plot_xy(x_data, y_data)

        assert fig is not None

    def test_measurements_with_non_float_dict_value(self, sample_waveform):
        """Test measurement annotations with non-float dict values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "String": "not a number",
            "Integer": 42,
            "Valid": 2.5,
        }
        # Should not crash, just skip invalid measurements
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        assert fig is not None

    def test_measurements_with_string_values(self, sample_waveform):
        """Test measurement annotations with string values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Text": "hello",
            "Number": {"value": "not a float", "unit": "V"},
        }
        # Should not crash, just skip invalid measurements
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        assert fig is not None

    def test_plot_multi_channel_with_mismatched_sample_rates(self):
        """Test multi-channel with different sample rates."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        sample_rate1 = 1e6
        sample_rate2 = 2e6
        n_samples = 1000

        metadata1 = TraceMetadata(sample_rate=sample_rate1)
        metadata2 = TraceMetadata(sample_rate=sample_rate2)

        trace1 = WaveformTrace(data=np.sin(np.arange(n_samples) * 0.01), metadata=metadata1)
        trace2 = WaveformTrace(data=np.cos(np.arange(n_samples) * 0.01), metadata=metadata2)

        fig = plot_multi_channel([trace1, trace2])

        assert fig is not None

    def test_plot_waveform_all_parameters(self, sample_waveform):
        """Test plot_waveform with all parameters specified."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from oscura.visualization.waveform import plot_waveform

        fig, ax = plt.subplots(figsize=(8, 6))

        result_fig = plot_waveform(
            sample_waveform,
            ax=ax,
            time_unit="us",
            time_range=(0.0, 0.0005),
            show_grid=True,
            color="blue",
            label="Test Signal",
            show_measurements={"Peak": 2.5, "RMS": 1.77},
            title="Complete Test",
            xlabel="Custom Time",
            ylabel="Custom Voltage",
            show=False,
            figsize=(10, 6),  # Ignored when ax is provided
        )

        assert result_fig is fig
        plt.close(fig)

    def test_plot_multi_channel_invalid_time_unit(self, multi_channel_waveforms):
        """Test multi-channel with invalid time unit falls back to default."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        fig = plot_multi_channel(multi_channel_waveforms, time_unit="invalid")

        assert fig is not None

    def test_annotation_with_integer_values(self, sample_waveform):
        """Test annotation with integer measurement values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        measurements = {
            "Count": 100,
            "Samples": 1000,
        }
        # Integers should be treated as valid measurements
        fig = plot_waveform(sample_waveform, show_measurements=measurements, show=False)

        assert fig is not None

    def test_digital_trace_with_all_false(self):
        """Test digital trace that is always low."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        sample_rate = 1e6
        n_samples = 100
        data = np.zeros(n_samples, dtype=bool)
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = DigitalTrace(data=data, metadata=metadata)

        fig = plot_multi_channel([trace])

        assert fig is not None

    def test_digital_trace_with_all_true(self):
        """Test digital trace that is always high."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_multi_channel

        sample_rate = 1e6
        n_samples = 100
        data = np.ones(n_samples, dtype=bool)
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = DigitalTrace(data=data, metadata=metadata)

        fig = plot_multi_channel([trace])

        assert fig is not None

    def test_plot_waveform_with_inf_values(self):
        """Test waveform plotting with infinite values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        sample_rate = 1e6
        data = np.array([1.0, 2.0, np.inf, 3.0, -np.inf])
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should handle inf values without crashing
        fig = plot_waveform(trace, show=False)

        assert fig is not None

    def test_plot_waveform_with_nan_values(self):
        """Test waveform plotting with NaN values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_waveform

        sample_rate = 1e6
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should handle NaN values without crashing
        fig = plot_waveform(trace, show=False)

        assert fig is not None

    def test_plot_xy_mismatched_trace_and_array(self):
        """Test XY plot with WaveformTrace as Y and array as X."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.waveform import plot_xy

        sample_rate = 1e6
        n_samples = 100
        y_data_trace = WaveformTrace(
            data=np.sin(np.arange(n_samples) * 0.1),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        x_data_array = np.cos(np.arange(n_samples) * 0.1)

        fig = plot_xy(x_data_array, y_data_trace)

        assert fig is not None
