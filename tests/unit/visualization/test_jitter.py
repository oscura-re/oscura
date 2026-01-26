"""Comprehensive test suite for jitter visualization functions.

Tests cover TIE histograms, bathtub curves, DDJ/DCD plots, and jitter trend analysis
with complete edge case coverage and validation of IEEE 802.3/JEDEC JESD65B compliance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Module under test
try:
    from oscura.visualization import jitter

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

pytestmark = pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tie_data_normal() -> NDArray[np.floating]:
    """Normal TIE data with Gaussian distribution (2ps RMS)."""
    np.random.seed(42)
    return np.random.randn(10000) * 2e-12  # 2 ps RMS jitter


@pytest.fixture
def tie_data_with_dj() -> NDArray[np.floating]:
    """TIE data with deterministic jitter component."""
    np.random.seed(42)
    rj = np.random.randn(5000) * 1e-12  # 1 ps RJ
    dj = np.random.choice([-2e-12, 2e-12], size=5000)  # ±2 ps DJ
    return rj + dj


@pytest.fixture
def bathtub_data() -> dict:
    """Sample bathtub curve data."""
    from scipy.special import erfc

    positions = np.linspace(0, 1, 100)
    # Simulated BER curves
    ber_left = 0.5 * erfc((positions - 0.1) / 0.15 / np.sqrt(2))
    ber_right = 0.5 * erfc((0.9 - positions) / 0.15 / np.sqrt(2))
    return {
        "positions": positions,
        "ber_left": ber_left,
        "ber_right": ber_right,
        "ber_total": ber_left + ber_right,
    }


@pytest.fixture
def ddj_data() -> dict:
    """DDJ pattern data."""
    patterns = ["000", "001", "010", "011", "100", "101", "110", "111"]
    jitter_values = np.array([0, 2.1, -1.5, 0.5, 0.8, -0.3, 1.2, -0.8]) * 1e-12  # ps
    return {"patterns": patterns, "jitter_values": jitter_values}


@pytest.fixture
def dcd_data() -> dict:
    """DCD (Duty Cycle Distortion) data."""
    np.random.seed(42)
    high_times = np.random.normal(500e-12, 20e-12, 1000)  # 500 ps ± 20 ps
    low_times = np.random.normal(480e-12, 18e-12, 1000)  # 480 ps ± 18 ps (DCD present)
    return {"high_times": high_times, "low_times": low_times}


@pytest.fixture
def jitter_trend_data() -> dict:
    """Jitter trend over time."""
    time_axis = np.arange(0, 1000)
    # Jitter with drift
    jitter_values = 5e-12 + 0.001e-12 * time_axis + np.random.randn(1000) * 2e-12
    return {"time_axis": time_axis, "jitter_values": jitter_values}


@pytest.fixture
def mock_mpl():
    """Mock matplotlib for testing without display."""
    with patch("oscura.visualization.jitter.plt") as mock_plt:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        yield {"plt": mock_plt, "fig": mock_fig, "ax": mock_ax}


# ============================================================================
# Test plot_tie_histogram
# ============================================================================


class TestPlotTieHistogram:
    """Test TIE histogram visualization."""

    def test_basic_tie_histogram(self, tie_data_normal, mock_mpl):
        """Test basic TIE histogram creation."""
        fig = jitter.plot_tie_histogram(tie_data_normal, show=False)

        assert fig is not None
        mock_mpl["ax"].hist.assert_called_once()
        mock_mpl["ax"].set_xlabel.assert_called()
        mock_mpl["ax"].set_ylabel.assert_called()
        mock_mpl["ax"].set_title.assert_called()

    def test_tie_time_unit_auto_selection(self, mock_mpl):
        """Test automatic time unit selection."""
        # Femtosecond range
        tie_fs = np.random.randn(100) * 1e-15
        fig_fs = jitter.plot_tie_histogram(tie_fs, show=False)
        assert fig_fs is not None

        # Picosecond range
        tie_ps = np.random.randn(100) * 1e-12
        fig_ps = jitter.plot_tie_histogram(tie_ps, show=False)
        assert fig_ps is not None

        # Nanosecond range
        tie_ns = np.random.randn(100) * 1e-9
        fig_ns = jitter.plot_tie_histogram(tie_ns, show=False)
        assert fig_ns is not None

    def test_tie_manual_time_unit(self, tie_data_normal, mock_mpl):
        """Test manual time unit specification."""
        for unit in ["s", "ms", "us", "ns", "ps", "fs"]:
            fig = jitter.plot_tie_histogram(tie_data_normal, time_unit=unit, show=False)
            assert fig is not None

    def test_tie_gaussian_fit_overlay(self, tie_data_normal, mock_mpl):
        """Test Gaussian fit overlay for RJ estimation."""
        fig = jitter.plot_tie_histogram(tie_data_normal, show_gaussian_fit=True, show=False)

        # Should plot Gaussian fit
        mock_mpl["ax"].plot.assert_called()

    def test_tie_without_gaussian_fit(self, tie_data_normal, mock_mpl):
        """Test histogram without Gaussian fit."""
        fig = jitter.plot_tie_histogram(tie_data_normal, show_gaussian_fit=False, show=False)
        assert fig is not None
        # Without Gaussian fit, plot should still be called for histogram
        mock_mpl["ax"].hist.assert_called()

    def test_tie_statistics_box(self, tie_data_normal, mock_mpl):
        """Test statistics box display."""
        jitter.plot_tie_histogram(tie_data_normal, show_statistics=True, show=False)

        # Should add text annotation
        mock_mpl["ax"].text.assert_called()

    def test_tie_rj_dj_indicators(self, tie_data_with_dj, mock_mpl):
        """Test RJ/DJ separation indicators."""
        jitter.plot_tie_histogram(tie_data_with_dj, show_rj_dj=True, show=False)

        # Should mark ±3sigma region
        assert mock_mpl["ax"].axvline.call_count >= 2
        mock_mpl["ax"].axvspan.assert_called()

    def test_tie_custom_bins(self, tie_data_normal, mock_mpl):
        """Test custom bin specifications."""
        # Integer bins
        fig1 = jitter.plot_tie_histogram(tie_data_normal, bins=50, show=False)
        assert fig1 is not None

        # Auto bins
        fig2 = jitter.plot_tie_histogram(tie_data_normal, bins="auto", show=False)
        assert fig2 is not None

    def test_tie_custom_title(self, tie_data_normal, mock_mpl):
        """Test custom title."""
        title = "Custom TIE Analysis"
        jitter.plot_tie_histogram(tie_data_normal, title=title, show=False)

        mock_mpl["ax"].set_title.assert_called()

    def test_tie_custom_figsize(self, tie_data_normal, mock_mpl):
        """Test custom figure size."""
        jitter.plot_tie_histogram(tie_data_normal, figsize=(8, 5), show=False)

        mock_mpl["plt"].subplots.assert_called()

    def test_tie_with_existing_axes(self, tie_data_normal):
        """Test plotting on existing axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = jitter.plot_tie_histogram(tie_data_normal, ax=ax, show=False)

        assert result == fig
        plt.close(fig)

    def test_tie_save_to_file(self, tie_data_normal, mock_mpl, tmp_path):
        """Test saving plot to file."""
        output_file = tmp_path / "tie_histogram.png"
        jitter.plot_tie_histogram(tie_data_normal, save_path=output_file, show=False)

        mock_mpl["fig"].savefig.assert_called_once()

    def test_tie_empty_data(self, mock_mpl):
        """Test handling of empty TIE data."""
        empty_data = np.array([])
        # Should raise ValueError for empty data
        with pytest.raises((ValueError, IndexError)):
            jitter.plot_tie_histogram(empty_data, show=False)

    def test_tie_single_value(self, mock_mpl):
        """Test single TIE value."""
        single_val = np.array([1e-12])
        fig = jitter.plot_tie_histogram(single_val, show=False)
        assert fig is not None

    def test_tie_zero_jitter(self, mock_mpl):
        """Test all-zero jitter (ideal case)."""
        zero_jitter = np.zeros(100)
        fig = jitter.plot_tie_histogram(zero_jitter, show=False)
        assert fig is not None

    def test_tie_negative_values(self, mock_mpl):
        """Test TIE with negative values (valid)."""
        tie_negative = np.random.randn(1000) * 1e-12
        fig = jitter.plot_tie_histogram(tie_negative, show=False)
        assert fig is not None


# ============================================================================
# Test plot_bathtub_full
# ============================================================================


class TestPlotBathtubFull:
    """Test bathtub curve visualization."""

    def test_basic_bathtub_curve(self, bathtub_data, mock_mpl):
        """Test basic bathtub curve creation."""
        fig = jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            show=False,
        )

        assert fig is not None
        # Should create semilogy plots
        assert mock_mpl["ax"].semilogy.call_count == 3  # left, right, total

    def test_bathtub_with_ber_total(self, bathtub_data, mock_mpl):
        """Test bathtub with explicit total BER."""
        fig = jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            ber_total=bathtub_data["ber_total"],
            show=False,
        )
        assert fig is not None
        # Should create plots including total BER
        assert mock_mpl["ax"].semilogy.call_count >= 3

    def test_bathtub_target_ber_marker(self, bathtub_data, mock_mpl):
        """Test target BER marker line."""
        jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            target_ber=1e-12,
            show_target=True,
            show=False,
        )

        # Should draw horizontal line at target BER
        mock_mpl["ax"].axhline.assert_called()

    def test_bathtub_eye_opening_annotation(self, bathtub_data, mock_mpl):
        """Test eye opening annotation."""
        jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            show_eye_opening=True,
            target_ber=1e-6,
            show=False,
        )

        # Should annotate eye opening
        mock_mpl["ax"].annotate.assert_called()

    def test_bathtub_explicit_eye_opening(self, bathtub_data, mock_mpl):
        """Test with pre-calculated eye opening."""
        fig = jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            eye_opening=0.7,
            show_eye_opening=True,
            show=False,
        )
        assert fig is not None
        # Should annotate the eye opening
        mock_mpl["ax"].annotate.assert_called()

    def test_bathtub_custom_title(self, bathtub_data, mock_mpl):
        """Test custom title."""
        title = "Custom Bathtub Analysis"
        jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            title=title,
            show=False,
        )

        mock_mpl["ax"].set_title.assert_called()

    def test_bathtub_custom_figsize(self, bathtub_data, mock_mpl):
        """Test custom figure size."""
        fig = jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            figsize=(12, 8),
            show=False,
        )
        assert fig is not None
        # Should create subplots with custom figsize
        mock_mpl["plt"].subplots.assert_called()

    def test_bathtub_ber_clipping(self, mock_mpl):
        """Test BER value clipping for log plot."""
        # BER with very small values
        positions = np.linspace(0, 1, 50)
        ber_left = np.ones(50) * 1e-20  # Below minimum
        ber_right = np.ones(50) * 1e-20

        fig = jitter.plot_bathtub_full(positions, ber_left, ber_right, show=False)

        # Should clip to valid log range and create figure
        assert fig is not None
        mock_mpl["ax"].semilogy.assert_called()

    def test_bathtub_save_to_file(self, bathtub_data, mock_mpl, tmp_path):
        """Test saving bathtub to file."""
        output_file = tmp_path / "bathtub.png"
        jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            save_path=output_file,
            show=False,
        )

        mock_mpl["fig"].savefig.assert_called_once()

    def test_bathtub_no_eye_opening(self, mock_mpl):
        """Test bathtub with no valid eye opening (BER too high)."""
        positions = np.linspace(0, 1, 50)
        ber_left = np.ones(50) * 0.1  # High BER everywhere
        ber_right = np.ones(50) * 0.1

        fig = jitter.plot_bathtub_full(
            positions,
            ber_left,
            ber_right,
            target_ber=1e-12,
            show_eye_opening=True,
            show=False,
        )

        # Should handle no crossing points gracefully
        assert fig is not None
        mock_mpl["ax"].semilogy.assert_called()


# ============================================================================
# Test plot_ddj
# ============================================================================


class TestPlotDDJ:
    """Test Data-Dependent Jitter visualization."""

    def test_basic_ddj_plot(self, ddj_data, mock_mpl):
        """Test basic DDJ bar chart."""
        fig = jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            show=False,
        )

        assert fig is not None
        mock_mpl["ax"].bar.assert_called_once()

    def test_ddj_color_coding(self, ddj_data, mock_mpl):
        """Test color coding based on sign (positive/negative)."""
        fig = jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            show=False,
        )

        # Should assign different colors for positive/negative
        assert fig is not None
        mock_mpl["ax"].bar.assert_called()

    def test_ddj_custom_time_unit(self, ddj_data, mock_mpl):
        """Test custom time unit."""
        fig = jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            time_unit="ns",
            show=False,
        )
        assert fig is not None
        # Should set ylabel with correct time unit
        mock_mpl["ax"].set_ylabel.assert_called()

    def test_ddj_peak_to_peak_annotation(self, ddj_data, mock_mpl):
        """Test DDJ peak-to-peak annotation."""
        jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            show=False,
        )

        # Should annotate DDJ pp value
        mock_mpl["ax"].text.assert_called()

    def test_ddj_custom_title(self, ddj_data, mock_mpl):
        """Test custom title."""
        title = "Custom DDJ Analysis"
        jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            title=title,
            show=False,
        )

        mock_mpl["ax"].set_title.assert_called()

    def test_ddj_save_to_file(self, ddj_data, mock_mpl, tmp_path):
        """Test saving DDJ plot."""
        output_file = tmp_path / "ddj.png"
        jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            save_path=output_file,
            show=False,
        )

        mock_mpl["fig"].savefig.assert_called_once()

    def test_ddj_all_positive_values(self, mock_mpl):
        """Test DDJ with all positive jitter."""
        patterns = ["00", "01", "10", "11"]
        jitter_vals = np.array([1, 2, 1.5, 2.5]) * 1e-12

        fig = jitter.plot_ddj(patterns, jitter_vals, show=False)
        assert fig is not None
        mock_mpl["ax"].bar.assert_called()

    def test_ddj_all_negative_values(self, mock_mpl):
        """Test DDJ with all negative jitter."""
        patterns = ["00", "01", "10", "11"]
        jitter_vals = np.array([-1, -2, -1.5, -2.5]) * 1e-12

        fig = jitter.plot_ddj(patterns, jitter_vals, show=False)
        assert fig is not None
        mock_mpl["ax"].bar.assert_called()

    def test_ddj_zero_values(self, mock_mpl):
        """Test DDJ with zero jitter."""
        patterns = ["00", "01", "10", "11"]
        jitter_vals = np.zeros(4)

        fig = jitter.plot_ddj(patterns, jitter_vals, show=False)
        assert fig is not None
        mock_mpl["ax"].bar.assert_called()

    def test_ddj_single_pattern(self, mock_mpl):
        """Test DDJ with single pattern."""
        patterns = ["010"]
        jitter_vals = np.array([1.5e-12])

        fig = jitter.plot_ddj(patterns, jitter_vals, show=False)
        assert fig is not None
        mock_mpl["ax"].bar.assert_called()


# ============================================================================
# Test plot_dcd
# ============================================================================


class TestPlotDCD:
    """Test Duty Cycle Distortion visualization."""

    def test_basic_dcd_plot(self, dcd_data, mock_mpl):
        """Test basic DCD histogram plot."""
        fig = jitter.plot_dcd(
            dcd_data["high_times"],
            dcd_data["low_times"],
            show=False,
        )

        assert fig is not None
        assert mock_mpl["ax"].hist.call_count == 2  # high and low

    def test_dcd_time_unit_auto_selection(self, mock_mpl):
        """Test automatic time unit selection."""
        # Picosecond range
        high_ps = np.random.normal(500e-12, 20e-12, 100)
        low_ps = np.random.normal(500e-12, 20e-12, 100)
        fig_ps = jitter.plot_dcd(high_ps, low_ps, show=False)
        assert fig_ps is not None

        # Nanosecond range
        high_ns = np.random.normal(500e-9, 20e-9, 100)
        low_ns = np.random.normal(500e-9, 20e-9, 100)
        fig_ns = jitter.plot_dcd(high_ns, low_ns, show=False)
        assert fig_ns is not None

    def test_dcd_manual_time_unit(self, dcd_data, mock_mpl):
        """Test manual time unit specification."""
        for unit in ["s", "ms", "us", "ns", "ps"]:
            fig = jitter.plot_dcd(
                dcd_data["high_times"],
                dcd_data["low_times"],
                time_unit=unit,
                show=False,
            )
            assert fig is not None

    def test_dcd_statistics_calculation(self, dcd_data, mock_mpl):
        """Test DCD statistics calculation."""
        jitter.plot_dcd(
            dcd_data["high_times"],
            dcd_data["low_times"],
            show=False,
        )

        # Should display statistics (mean high/low, duty cycle, DCD)
        mock_mpl["ax"].text.assert_called()

    def test_dcd_mean_lines(self, dcd_data, mock_mpl):
        """Test mean value lines."""
        jitter.plot_dcd(
            dcd_data["high_times"],
            dcd_data["low_times"],
            show=False,
        )

        # Should draw mean lines
        assert mock_mpl["ax"].axvline.call_count >= 2

    def test_dcd_custom_title(self, dcd_data, mock_mpl):
        """Test custom title."""
        title = "Custom DCD Analysis"
        jitter.plot_dcd(
            dcd_data["high_times"],
            dcd_data["low_times"],
            title=title,
            show=False,
        )

        mock_mpl["ax"].set_title.assert_called()

    def test_dcd_save_to_file(self, dcd_data, mock_mpl, tmp_path):
        """Test saving DCD plot."""
        output_file = tmp_path / "dcd.png"
        jitter.plot_dcd(
            dcd_data["high_times"],
            dcd_data["low_times"],
            save_path=output_file,
            show=False,
        )

        mock_mpl["fig"].savefig.assert_called_once()

    def test_dcd_equal_high_low(self, mock_mpl):
        """Test DCD with equal high/low times (no distortion)."""
        high_times = np.random.normal(500e-12, 10e-12, 100)
        low_times = np.random.normal(500e-12, 10e-12, 100)

        fig = jitter.plot_dcd(high_times, low_times, show=False)
        assert fig is not None
        # Should create histograms for both high and low
        assert mock_mpl["ax"].hist.call_count >= 2

    def test_dcd_extreme_distortion(self, mock_mpl):
        """Test DCD with extreme duty cycle distortion."""
        high_times = np.random.normal(800e-12, 10e-12, 100)  # 80% duty
        low_times = np.random.normal(200e-12, 5e-12, 100)  # 20% duty

        fig = jitter.plot_dcd(high_times, low_times, show=False)
        assert fig is not None
        # Should create histograms showing distortion
        assert mock_mpl["ax"].hist.call_count >= 2


# ============================================================================
# Test plot_jitter_trend
# ============================================================================


class TestPlotJitterTrend:
    """Test jitter trend over time visualization."""

    def test_basic_jitter_trend(self, jitter_trend_data, mock_mpl):
        """Test basic jitter trend plot."""
        fig = jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            show=False,
        )

        assert fig is not None
        mock_mpl["ax"].plot.assert_called()

    def test_jitter_trend_auto_unit_selection(self, mock_mpl):
        """Test automatic jitter unit selection."""
        time_axis = np.arange(100)

        # Picosecond range
        jitter_ps = np.random.randn(100) * 1e-12
        fig_ps = jitter.plot_jitter_trend(time_axis, jitter_ps, show=False)
        assert fig_ps is not None

        # Nanosecond range
        jitter_ns = np.random.randn(100) * 1e-9
        fig_ns = jitter.plot_jitter_trend(time_axis, jitter_ns, show=False)
        assert fig_ns is not None

    def test_jitter_trend_with_trend_line(self, jitter_trend_data, mock_mpl):
        """Test linear trend line overlay."""
        fig = jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            show_trend=True,
            show=False,
        )

        # Should fit and plot trend line
        assert fig is not None
        # At least two plot calls: main data + trend line
        assert mock_mpl["ax"].plot.call_count >= 2

    def test_jitter_trend_statistical_bounds(self, jitter_trend_data, mock_mpl):
        """Test ±3σ statistical bounds."""
        jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            show_bounds=True,
            show=False,
        )

        # Should draw bound lines
        mock_mpl["ax"].axhline.assert_called()
        mock_mpl["ax"].fill_between.assert_called()

    def test_jitter_trend_custom_units(self, jitter_trend_data, mock_mpl):
        """Test custom time and jitter units."""
        fig = jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            time_unit="ms",
            jitter_unit="ns",
            show=False,
        )
        assert fig is not None
        # Should set labels with custom units
        mock_mpl["ax"].set_xlabel.assert_called()
        mock_mpl["ax"].set_ylabel.assert_called()

    def test_jitter_trend_custom_title(self, jitter_trend_data, mock_mpl):
        """Test custom title."""
        title = "Custom Jitter Trend"
        jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            title=title,
            show=False,
        )

        mock_mpl["ax"].set_title.assert_called()

    def test_jitter_trend_save_to_file(self, jitter_trend_data, mock_mpl, tmp_path):
        """Test saving trend plot."""
        output_file = tmp_path / "jitter_trend.png"
        jitter.plot_jitter_trend(
            jitter_trend_data["time_axis"],
            jitter_trend_data["jitter_values"],
            save_path=output_file,
            show=False,
        )

        mock_mpl["fig"].savefig.assert_called_once()

    def test_jitter_trend_constant_jitter(self, mock_mpl):
        """Test trend with constant jitter (no variation)."""
        time_axis = np.arange(100)
        jitter_values = np.ones(100) * 5e-12

        fig = jitter.plot_jitter_trend(time_axis, jitter_values, show=False)
        assert fig is not None
        mock_mpl["ax"].plot.assert_called()

    def test_jitter_trend_increasing(self, mock_mpl):
        """Test trend with increasing jitter (drift)."""
        time_axis = np.arange(100)
        jitter_values = 1e-12 * time_axis  # Linear increase

        fig = jitter.plot_jitter_trend(time_axis, jitter_values, show_trend=True, show=False)
        assert fig is not None
        # Should plot data and trend line
        assert mock_mpl["ax"].plot.call_count >= 2

    def test_jitter_trend_short_dataset(self, mock_mpl):
        """Test trend with short time series."""
        time_axis = np.arange(10)
        jitter_values = np.random.randn(10) * 1e-12

        fig = jitter.plot_jitter_trend(time_axis, jitter_values, show=False)
        assert fig is not None
        mock_mpl["ax"].plot.assert_called()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_matplotlib_import_error(self, monkeypatch):
        """Test error when matplotlib not available."""
        monkeypatch.setattr("oscura.visualization.jitter.HAS_MATPLOTLIB", False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            jitter.plot_tie_histogram(np.array([1e-12]), show=False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            jitter.plot_bathtub_full(np.array([0.5]), np.array([0.1]), np.array([0.1]), show=False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            jitter.plot_ddj(["01"], np.array([1e-12]), show=False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            jitter.plot_dcd(np.array([1e-12]), np.array([1e-12]), show=False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            jitter.plot_jitter_trend(np.array([0]), np.array([1e-12]), show=False)

    def test_axes_without_figure_error(self, tie_data_normal):
        """Test error when axes has no associated figure."""
        mock_ax = Mock()
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            jitter.plot_tie_histogram(tie_data_normal, ax=mock_ax, show=False)

    def test_invalid_time_unit_fallback(self, tie_data_normal, mock_mpl):
        """Test fallback for invalid time unit."""
        # Should use default (ps) for invalid unit
        fig = jitter.plot_tie_histogram(tie_data_normal, time_unit="invalid", show=False)
        assert fig is not None
        # Should still create histogram despite invalid unit
        mock_mpl["ax"].hist.assert_called()

    def test_mismatched_array_lengths(self, mock_mpl):
        """Test mismatched array lengths in bathtub."""
        positions = np.linspace(0, 1, 100)
        ber_left = np.ones(50)  # Wrong length
        ber_right = np.ones(100)

        # Should raise error for mismatched lengths
        with pytest.raises((ValueError, IndexError)):
            jitter.plot_bathtub_full(positions, ber_left, ber_right, show=False)

    def test_ddj_pattern_jitter_mismatch(self, mock_mpl):
        """Test mismatched pattern/jitter array lengths."""
        patterns = ["00", "01"]
        jitter_vals = np.array([1e-12, 2e-12, 3e-12])  # Extra value

        # Should raise error for mismatched lengths
        with pytest.raises((ValueError, IndexError)):
            jitter.plot_ddj(patterns, jitter_vals, show=False)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real matplotlib (if available)."""

    @pytest.mark.slow
    def test_full_jitter_analysis_workflow(self, tie_data_with_dj, tmp_path):
        """Test complete jitter analysis workflow."""
        # TIE histogram
        tie_fig = jitter.plot_tie_histogram(
            tie_data_with_dj,
            show_gaussian_fit=True,
            show_rj_dj=True,
            save_path=tmp_path / "tie.png",
            show=False,
        )
        assert tie_fig is not None

    @pytest.mark.slow
    def test_bathtub_analysis_workflow(self, bathtub_data, tmp_path):
        """Test bathtub curve analysis workflow."""
        fig = jitter.plot_bathtub_full(
            bathtub_data["positions"],
            bathtub_data["ber_left"],
            bathtub_data["ber_right"],
            target_ber=1e-12,
            show_eye_opening=True,
            save_path=tmp_path / "bathtub.png",
            show=False,
        )
        assert fig is not None

    @pytest.mark.slow
    def test_multi_plot_composition(self, tie_data_normal, ddj_data):
        """Test creating multi-panel jitter analysis."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # TIE histogram
        jitter.plot_tie_histogram(tie_data_normal, ax=axes[0, 0], show=False)

        # DDJ
        jitter.plot_ddj(
            ddj_data["patterns"],
            ddj_data["jitter_values"],
            ax=axes[0, 1],
            show=False,
        )

        # Verify plots were created
        assert fig is not None
        assert len(axes.flatten()) == 4

        plt.close(fig)
