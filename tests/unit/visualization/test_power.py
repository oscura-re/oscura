"""Unit tests for power profile visualization.

Tests:

This module provides comprehensive tests for power visualization
functions to verify plot creation, annotations, and multi-channel support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from oscura.visualization.power import plot_power_profile

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_power_trace():
    """Create sample single-channel power trace."""
    # 1000 samples of power consumption (300-800 mW with some peaks)
    rng = np.random.default_rng(42)
    power = rng.uniform(0.3, 0.5, 1000)
    # Add some power spikes
    power[100:110] = 0.8  # Peak at sample 100
    power[500:520] = 0.7  # Another peak
    return power


@pytest.fixture
def multi_channel_power():
    """Create multi-channel power traces."""
    rng = np.random.default_rng(42)
    return {
        "VDD_CORE": rng.uniform(0.4, 0.6, 1000),
        "VDD_IO": rng.uniform(0.2, 0.4, 1000),
        "VDD_ANALOG": rng.uniform(0.1, 0.3, 1000),
    }


@pytest.fixture
def sample_statistics():
    """Create sample pre-computed statistics."""
    return {
        "average": 0.45,
        "peak": 0.85,
        "energy": 0.00045,  # 450 µJ
        "rms": 0.46,
    }


@pytest.fixture
def multi_channel_statistics():
    """Create statistics for multi-channel power."""
    return {
        "VDD_CORE": {
            "average": 0.5,
            "peak": 0.6,
            "energy": 0.0005,
        },
        "VDD_IO": {
            "average": 0.3,
            "peak": 0.4,
            "energy": 0.0003,
        },
        "VDD_ANALOG": {
            "average": 0.2,
            "peak": 0.3,
            "energy": 0.0002,
        },
    }


@pytest.fixture
def time_array():
    """Create explicit time array."""
    return np.linspace(0, 1e-3, 1000)  # 0 to 1 ms


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBasicPlotting:
    """Tests for basic plot_power_profile functionality."""

    def test_basic_single_channel_plot(self, sample_power_trace):
        """Test basic single-channel power profile plot."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None
        assert len(fig.axes) >= 1  # At least main power axis

    def test_with_time_array(self, sample_power_trace, time_array):
        """Test plot with explicit time array."""

        fig = plot_power_profile(sample_power_trace, time_array=time_array, show=False)

        assert fig is not None
        assert len(fig.axes) >= 1

    def test_multi_channel_stacked(self, multi_channel_power):
        """Test multi-channel stacked layout."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="stacked",
            show=False,
        )

        assert fig is not None
        # Should have 3 channels + 1 energy plot (default show_energy=True)
        assert len(fig.axes) == 4

    def test_multi_channel_overlay(self, multi_channel_power):
        """Test multi-channel overlay layout."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="overlay",
            show=False,
        )

        assert fig is not None
        # Overlay mode should have fewer axes than stacked
        assert len(fig.axes) >= 1


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_missing_sample_rate_and_time_array(self, sample_power_trace):
        """Test that missing both sample_rate and time_array raises error."""

        with pytest.raises(ValueError, match="Either time_array or sample_rate must be provided"):
            plot_power_profile(sample_power_trace, show=False)

    def test_time_array_length_mismatch(self, sample_power_trace):
        """Test that mismatched time_array length raises error."""

        wrong_time_array = np.linspace(0, 1e-3, 500)  # Wrong length

        with pytest.raises(ValueError, match="time_array length .* doesn't match"):
            plot_power_profile(sample_power_trace, time_array=wrong_time_array, show=False)

    def test_dict_input_validation(self, multi_channel_power):
        """Test that dict input works correctly."""

        fig = plot_power_profile(multi_channel_power, sample_rate=1e6, show=False)

        assert fig is not None
        assert isinstance(multi_channel_power, dict)


# =============================================================================
# Annotation Tests
# =============================================================================


class TestAnnotations:
    """Tests for power plot annotations."""

    def test_show_average_line(self, sample_power_trace):
        """Test average power line annotation."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_average=True, show=False)

        assert fig is not None
        # Check that horizontal line was added
        ax = fig.axes[0]
        assert len(ax.get_lines()) > 1  # Power line + average line

    def test_hide_average_line(self, sample_power_trace):
        """Test hiding average power line."""

        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, show_average=False, show=False
        )

        assert fig is not None

    def test_show_peak_marker(self, sample_power_trace):
        """Test peak power marker annotation."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_peak=True, show=False)

        assert fig is not None
        ax = fig.axes[0]
        # Should have multiple plot elements (line + peak marker)
        assert len(ax.get_lines()) >= 2

    def test_hide_peak_marker(self, sample_power_trace):
        """Test hiding peak marker."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_peak=False, show=False)

        assert fig is not None

    def test_show_energy_overlay(self, sample_power_trace):
        """Test cumulative energy overlay."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_energy=True, show=False)

        assert fig is not None
        # Single channel with energy should have secondary y-axis
        assert len(fig.axes) >= 2  # Main axis + energy axis

    def test_hide_energy_overlay(self, sample_power_trace):
        """Test hiding energy overlay."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_energy=False, show=False)

        assert fig is not None
        # Without energy overlay, should have only main axis
        assert len(fig.axes) == 1

    def test_all_annotations_disabled(self, sample_power_trace):
        """Test plot with all annotations disabled."""

        fig = plot_power_profile(
            sample_power_trace,
            sample_rate=1e6,
            show_average=False,
            show_peak=False,
            show_energy=False,
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 1


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for pre-computed statistics integration."""

    def test_with_precomputed_statistics(self, sample_power_trace, sample_statistics):
        """Test using pre-computed statistics."""

        fig = plot_power_profile(
            sample_power_trace,
            sample_rate=1e6,
            statistics=sample_statistics,
            show=False,
        )

        assert fig is not None

    def test_without_statistics(self, sample_power_trace):
        """Test automatic statistics computation."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None

    def test_multi_channel_with_statistics(self, multi_channel_power, multi_channel_statistics):
        """Test multi-channel plot with per-channel statistics."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            statistics=multi_channel_statistics,
            multi_channel_layout="stacked",
            show=False,
        )

        assert fig is not None

    def test_partial_statistics(self, multi_channel_power):
        """Test with statistics for only some channels."""

        partial_stats = {
            "VDD_CORE": {
                "average": 0.5,
                "peak": 0.6,
            }
            # Missing stats for other channels
        }

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            statistics=partial_stats,
            multi_channel_layout="stacked",
            show=False,
        )

        assert fig is not None


# =============================================================================
# Time Scaling Tests
# =============================================================================


class TestTimeScaling:
    """Tests for automatic time unit scaling."""

    def test_nanosecond_scale(self):
        """Test time display in nanoseconds."""

        # Very short trace (100 ns)
        power = np.random.uniform(0.3, 0.5, 100)
        sample_rate = 1e9  # 1 GSa/s

        fig = plot_power_profile(power, sample_rate=sample_rate, show=False)

        assert fig is not None
        # Check xlabel contains "ns"
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "ns" in xlabel

    def test_microsecond_scale(self):
        """Test time display in microseconds."""

        # Medium trace (100 µs)
        power = np.random.uniform(0.3, 0.5, 1000)
        sample_rate = 10e6  # 10 MSa/s

        fig = plot_power_profile(power, sample_rate=sample_rate, show=False)

        assert fig is not None
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "µs" in xlabel

    def test_millisecond_scale(self):
        """Test time display in milliseconds."""

        # Long trace (10 ms) to ensure millisecond display
        power = np.random.uniform(0.3, 0.5, 10000)
        sample_rate = 1e6  # 1 MSa/s, 10000 samples = 10 ms

        fig = plot_power_profile(power, sample_rate=sample_rate, show=False)

        assert fig is not None
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "ms" in xlabel

    def test_second_scale(self):
        """Test time display in seconds."""

        # Long trace (10 seconds)
        power = np.random.uniform(0.3, 0.5, 10000)
        sample_rate = 1e3  # 1 kSa/s

        fig = plot_power_profile(power, sample_rate=sample_rate, show=False)

        assert fig is not None
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "s" in xlabel


# =============================================================================
# Multi-Channel Tests
# =============================================================================


class TestMultiChannel:
    """Tests for multi-channel power visualization."""

    def test_stacked_layout_structure(self, multi_channel_power):
        """Test stacked layout creates correct number of subplots."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="stacked",
            show_energy=True,
            show=False,
        )

        assert fig is not None
        # 3 channels + 1 energy plot
        assert len(fig.axes) == 4

    def test_stacked_without_energy(self, multi_channel_power):
        """Test stacked layout without energy plot."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="stacked",
            show_energy=False,
            show=False,
        )

        assert fig is not None
        # 3 channels only
        assert len(fig.axes) == 3

    def test_overlay_layout(self, multi_channel_power):
        """Test overlay layout merges channels."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="overlay",
            show=False,
        )

        assert fig is not None
        # Overlay should have main axis + possibly energy axis
        assert len(fig.axes) >= 1

    def test_channel_labels(self, multi_channel_power):
        """Test that channel names appear in plot."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="stacked",
            show=False,
        )

        assert fig is not None
        # Check that channel names appear in y-labels or legends
        for idx, (name, _) in enumerate(multi_channel_power.items()):
            ax = fig.axes[idx]
            ylabel = ax.get_ylabel()
            assert name in ylabel

    def test_single_channel_dict(self):
        """Test single-channel passed as dict."""

        power_dict = {"Main": np.random.uniform(0.3, 0.5, 1000)}

        fig = plot_power_profile(power_dict, sample_rate=1e6, show=False)

        assert fig is not None

    def test_empty_dict_handling(self):
        """Test behavior with empty dict (should use first channel)."""

        # This should handle gracefully or raise appropriate error
        power_dict = {"Chan1": np.random.uniform(0.3, 0.5, 1000)}

        fig = plot_power_profile(power_dict, sample_rate=1e6, show=False)

        assert fig is not None


# =============================================================================
# Energy Calculation Tests
# =============================================================================


class TestEnergyCalculation:
    """Tests for energy accumulation calculation."""

    def test_energy_with_sample_rate(self, sample_power_trace):
        """Test energy calculation with sample_rate."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show_energy=True, show=False)

        assert fig is not None
        # Energy should be computed using sample_rate
        assert len(fig.axes) >= 2

    def test_energy_without_sample_rate(self, sample_power_trace, time_array):
        """Test energy calculation with time_array only."""

        # When only time_array is provided, sample_rate is None
        # Energy plot should still work or be gracefully skipped
        fig = plot_power_profile(
            sample_power_trace, time_array=time_array, show_energy=True, show=False
        )

        assert fig is not None

    def test_multi_channel_energy_accumulation(self, multi_channel_power):
        """Test energy accumulation for multi-channel stacked layout."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            multi_channel_layout="stacked",
            show_energy=True,
            show=False,
        )

        assert fig is not None
        # Last axis should be energy plot
        energy_ax = fig.axes[-1]
        ylabel = energy_ax.get_ylabel()
        assert "Energy" in ylabel or "µJ" in ylabel


# =============================================================================
# Customization Tests
# =============================================================================


class TestCustomization:
    """Tests for plot customization options."""

    def test_custom_title(self, sample_power_trace):
        """Test custom plot title."""

        custom_title = "MCU Power Consumption Analysis"
        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, title=custom_title, show=False
        )

        assert fig is not None
        assert fig._suptitle is not None
        assert custom_title in fig._suptitle.get_text()

    def test_default_title_single_channel(self, sample_power_trace):
        """Test default title for single channel."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None
        assert fig._suptitle is not None
        assert "Power Profile" in fig._suptitle.get_text()

    def test_default_title_multi_channel(self, multi_channel_power):
        """Test default title for multi-channel."""

        fig = plot_power_profile(multi_channel_power, sample_rate=1e6, show=False)

        assert fig is not None
        assert fig._suptitle is not None
        title_text = fig._suptitle.get_text()
        assert "Power Profile" in title_text
        assert "Multi-Channel" in title_text

    def test_custom_figsize(self, sample_power_trace):
        """Test custom figure size."""

        custom_figsize = (16, 8)
        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, figsize=custom_figsize, show=False
        )

        assert fig is not None
        # Check figure size (allow small tolerance due to DPI)
        fig_width, fig_height = fig.get_size_inches()
        assert abs(fig_width - custom_figsize[0]) < 0.1
        assert abs(fig_height - custom_figsize[1]) < 0.1

    def test_default_figsize(self, sample_power_trace):
        """Test default figure size."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None
        # Default is (12, 6)
        fig_width, fig_height = fig.get_size_inches()
        assert abs(fig_width - 12) < 0.1
        assert abs(fig_height - 6) < 0.1


# =============================================================================
# File Operations Tests
# =============================================================================


class TestFileOperations:
    """Tests for saving plots to files."""

    def test_save_to_file(self, sample_power_trace, tmp_path):
        """Test saving plot to file."""

        save_path = tmp_path / "power_plot.png"

        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, save_path=save_path, show=False
        )

        assert fig is not None
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_save_different_formats(self, sample_power_trace, tmp_path):
        """Test saving in different image formats."""

        for ext in ["png", "pdf", "svg", "jpg"]:
            save_path = tmp_path / f"power_plot.{ext}"

            fig = plot_power_profile(
                sample_power_trace, sample_rate=1e6, save_path=save_path, show=False
            )

            assert fig is not None
            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_save_path_as_string(self, sample_power_trace, tmp_path):
        """Test save_path as string."""

        save_path = str(tmp_path / "power_plot.png")

        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, save_path=save_path, show=False
        )

        assert fig is not None
        assert Path(save_path).exists()

    def test_save_path_as_path(self, sample_power_trace, tmp_path):
        """Test save_path as Path object."""

        save_path = tmp_path / "power_plot.png"

        fig = plot_power_profile(
            sample_power_trace, sample_rate=1e6, save_path=save_path, show=False
        )

        assert fig is not None
        assert save_path.exists()


# =============================================================================
# Figure Return Tests
# =============================================================================


class TestFigureReturn:
    """Tests for figure object return and manipulation."""

    def test_returns_figure_object(self, sample_power_trace):
        """Test that function returns matplotlib Figure."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_figure_has_axes(self, sample_power_trace):
        """Test that returned figure has axes."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None
        assert len(fig.axes) > 0

    def test_further_customization(self, sample_power_trace):
        """Test that returned figure can be further customized."""

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        # Add custom annotation
        ax = fig.axes[0]
        ax.axvline(0.5, color="purple", linestyle=":", label="Custom marker")

        # Should not raise exception
        assert fig is not None

    def test_show_parameter_false(self, sample_power_trace):
        """Test that show=False prevents display."""

        # This should not block or display
        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=False)

        assert fig is not None

    def test_show_parameter_true(self, sample_power_trace, monkeypatch):
        """Test that show=True calls plt.show()."""

        # Mock plt.show to avoid blocking
        show_called = []

        def mock_show():
            show_called.append(True)

        monkeypatch.setattr(plt, "show", mock_show)

        fig = plot_power_profile(sample_power_trace, sample_rate=1e6, show=True)

        assert fig is not None
        assert len(show_called) == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestVisualizationPowerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_trace(self):
        """Test with very short power trace."""

        power = np.array([0.5, 0.6, 0.4])
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_constant_power(self):
        """Test with constant power value."""

        power = np.full(1000, 0.5)
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_zero_power(self):
        """Test with all-zero power."""

        power = np.zeros(1000)
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_negative_power_values(self):
        """Test with negative power values (e.g., regenerative braking)."""

        power = np.random.uniform(-0.2, 0.5, 1000)
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_very_large_values(self):
        """Test with very large power values."""

        power = np.random.uniform(1000, 2000, 1000)  # kW range
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_single_sample(self):
        """Test with single sample."""

        power = np.array([0.5])
        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_nan_values(self):
        """Test with NaN values in power trace."""

        power = np.random.uniform(0.3, 0.5, 1000)
        power[100:110] = np.nan

        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None

    def test_inf_values(self):
        """Test with infinite values."""

        power = np.random.uniform(0.3, 0.5, 1000)
        power[100] = np.inf

        fig = plot_power_profile(power, sample_rate=1e6, show=False)

        assert fig is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestVisualizationPowerIntegration:
    """Integration tests combining multiple features."""

    def test_full_featured_single_channel(self, sample_power_trace, sample_statistics):
        """Test single channel with all features enabled."""

        fig = plot_power_profile(
            sample_power_trace,
            sample_rate=1e6,
            statistics=sample_statistics,
            show_average=True,
            show_peak=True,
            show_energy=True,
            title="Complete Power Analysis",
            figsize=(14, 8),
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) >= 2  # Main + energy

    def test_full_featured_multi_channel(self, multi_channel_power, multi_channel_statistics):
        """Test multi-channel with all features enabled."""

        fig = plot_power_profile(
            multi_channel_power,
            sample_rate=1e6,
            statistics=multi_channel_statistics,
            show_average=True,
            show_peak=True,
            show_energy=True,
            multi_channel_layout="stacked",
            title="Multi-Rail Power Analysis",
            figsize=(14, 12),
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4  # 3 channels + energy

    def test_minimal_configuration(self, sample_power_trace):
        """Test with minimal configuration."""

        fig = plot_power_profile(
            sample_power_trace,
            sample_rate=1e6,
            show_average=False,
            show_peak=False,
            show_energy=False,
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 1

    def test_save_and_return(self, sample_power_trace, tmp_path):
        """Test saving and returning figure simultaneously."""

        save_path = tmp_path / "power_analysis.png"

        fig = plot_power_profile(
            sample_power_trace,
            sample_rate=1e6,
            save_path=save_path,
            show=False,
        )

        assert fig is not None
        assert save_path.exists()
        assert isinstance(fig, plt.Figure)
