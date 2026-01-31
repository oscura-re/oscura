"""Comprehensive unit tests for reporting.visualization module.

Tests for PlotStyler and IEEEPlotGenerator with full coverage of:
- All plot generation functions (waveform, FFT, PSD, spectrogram, eye diagram, histogram, jitter, power)
- Error handling for invalid inputs
- Edge cases and boundary conditions
- IEEE styling application
- Figure to base64 conversion
"""

from __future__ import annotations

import base64

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

from oscura.reporting.visualization import (
    HAS_MATPLOTLIB as MODULE_HAS_MATPLOTLIB,
)
from oscura.reporting.visualization import (
    IEEE_COLORS,
    IEEEPlotGenerator,
    PlotStyler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("cleanup_matplotlib"),
]

skip_if_no_matplotlib = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")


class TestIEEEColors:
    """Test IEEE color scheme constants."""

    def test_ieee_colors_defined(self):
        """Test that all required IEEE colors are defined."""
        required_colors = [
            "primary",
            "secondary",
            "accent",
            "success",
            "warning",
            "danger",
            "grid",
            "text",
        ]

        for color_name in required_colors:
            assert color_name in IEEE_COLORS
            assert isinstance(IEEE_COLORS[color_name], str)
            # Verify it's a hex color
            assert IEEE_COLORS[color_name].startswith("#")
            assert len(IEEE_COLORS[color_name]) == 7  # #RRGGBB format


@skip_if_no_matplotlib
class TestPlotStyler:
    """Test PlotStyler class for IEEE-compliant styling."""

    def test_apply_ieee_style_basic(self):
        """Test basic IEEE style application."""
        fig, ax = plt.subplots()
        styler = PlotStyler()

        styler.apply_ieee_style(ax, "Time (s)", "Voltage (V)", "Test Plot")

        # Verify labels are set
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Voltage (V)"
        assert ax.get_title() == "Test Plot"

        plt.close(fig)

    def test_apply_ieee_style_empty_labels(self):
        """Test style application with empty labels."""
        fig, ax = plt.subplots()
        styler = PlotStyler()

        styler.apply_ieee_style(ax, "", "", "")

        # Should not crash, labels should be empty
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert ax.get_title() == ""

        plt.close(fig)

    def test_apply_ieee_style_with_grid(self):
        """Test grid application."""
        fig, ax = plt.subplots()
        styler = PlotStyler()

        styler.apply_ieee_style(ax, grid=True)

        # Grid should be enabled (check via grid properties)
        assert ax.xaxis.get_gridlines()[0].get_visible()

        plt.close(fig)

    def test_apply_ieee_style_without_grid(self):
        """Test style without grid."""
        fig, ax = plt.subplots()
        styler = PlotStyler()

        styler.apply_ieee_style(ax, grid=False)

        # Grid should be disabled (check via grid properties)
        assert not ax.xaxis.get_gridlines()[0].get_visible()

        plt.close(fig)

    def test_apply_ieee_style_spine_styling(self):
        """Test that spines are styled correctly."""
        fig, ax = plt.subplots()
        styler = PlotStyler()

        styler.apply_ieee_style(ax)

        # Check spine colors
        for spine in ax.spines.values():
            assert spine.get_edgecolor() == matplotlib.colors.to_rgba(IEEE_COLORS["text"])

        plt.close(fig)

    def test_static_method(self):
        """Test that apply_ieee_style is a static method."""
        fig, ax = plt.subplots()

        # Should work without instantiation
        PlotStyler.apply_ieee_style(ax, "X", "Y", "Title")

        assert ax.get_xlabel() == "X"

        plt.close(fig)


@skip_if_no_matplotlib
class TestIEEEPlotGeneratorInit:
    """Test IEEEPlotGenerator initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        generator = IEEEPlotGenerator()

        assert generator.dpi == 150
        assert generator.figsize == (10, 6)
        assert isinstance(generator.styler, PlotStyler)

    def test_init_custom_dpi(self):
        """Test initialization with custom DPI."""
        generator = IEEEPlotGenerator(dpi=300)

        assert generator.dpi == 300
        assert generator.figsize == (10, 6)

    def test_init_custom_figsize(self):
        """Test initialization with custom figure size."""
        generator = IEEEPlotGenerator(figsize=(8, 4))

        assert generator.dpi == 150
        assert generator.figsize == (8, 4)

    def test_init_both_custom(self):
        """Test initialization with both custom parameters."""
        generator = IEEEPlotGenerator(dpi=200, figsize=(12, 8))

        assert generator.dpi == 200
        assert generator.figsize == (12, 8)


@skip_if_no_matplotlib
class TestPlotWaveform:
    """Test waveform plotting functionality."""

    def test_plot_waveform_basic(self):
        """Test basic waveform plotting."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)

        fig = generator.plot_waveform(t, signal)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_waveform_custom_labels(self):
        """Test waveform with custom labels."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        fig = generator.plot_waveform(
            t, signal, title="Custom Title", xlabel="Custom X", ylabel="Custom Y"
        )

        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"

        plt.close(fig)

    def test_plot_waveform_with_markers(self):
        """Test waveform with markers."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)
        markers = {"Event 1": 0.25, "Event 2": 0.75}

        fig = generator.plot_waveform(t, signal, markers=markers)

        # Should have vertical lines for markers
        assert fig is not None

        plt.close(fig)

    def test_plot_waveform_empty_signal(self):
        """Test waveform with empty arrays."""
        generator = IEEEPlotGenerator()
        t = np.array([])
        signal = np.array([])

        fig = generator.plot_waveform(t, signal)

        # Should create figure without crashing
        assert fig is not None

        plt.close(fig)

    def test_plot_waveform_single_point(self):
        """Test waveform with single data point."""
        generator = IEEEPlotGenerator()
        t = np.array([0.5])
        signal = np.array([1.0])

        fig = generator.plot_waveform(t, signal)

        assert fig is not None

        plt.close(fig)

    def test_plot_waveform_mismatched_lengths(self):
        """Test waveform with mismatched time and signal lengths."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100)
        signal = np.linspace(0, 1, 50)  # Different length

        # Should raise ValueError from matplotlib
        with pytest.raises((ValueError, RuntimeError)):
            generator.plot_waveform(t, signal)


@skip_if_no_matplotlib
class TestPlotFFT:
    """Test FFT magnitude spectrum plotting."""

    def test_plot_fft_basic(self):
        """Test basic FFT plotting."""
        generator = IEEEPlotGenerator()
        frequencies = np.fft.rfftfreq(1000, 1 / 1000)
        magnitude_db = np.random.randn(len(frequencies)) * 10 - 20

        fig = generator.plot_fft(frequencies, magnitude_db)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_fft_no_peak_markers(self):
        """Test FFT without peak markers."""
        generator = IEEEPlotGenerator()
        frequencies = np.fft.rfftfreq(1000, 1 / 1000)
        magnitude_db = np.random.randn(len(frequencies)) * 10 - 20

        fig = generator.plot_fft(frequencies, magnitude_db, peak_markers=0)

        assert fig is not None

        plt.close(fig)

    def test_plot_fft_with_peaks(self):
        """Test FFT with peak markers."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        magnitude_db = np.zeros(100)
        # Add peaks
        magnitude_db[20] = 10.0
        magnitude_db[50] = 15.0
        magnitude_db[80] = 12.0

        fig = generator.plot_fft(frequencies, magnitude_db, peak_markers=3)

        assert fig is not None

        plt.close(fig)

    def test_plot_fft_log_scale(self):
        """Test FFT with log scale for wide frequency range."""
        generator = IEEEPlotGenerator()
        # Wide frequency range (>2 decades) should trigger log scale
        frequencies = np.logspace(0, 3, 100)  # 1 Hz to 1 kHz
        magnitude_db = np.random.randn(len(frequencies)) * 10 - 20

        fig = generator.plot_fft(frequencies, magnitude_db)

        ax = fig.axes[0]
        # Should use log scale
        assert ax.get_xscale() == "log"

        plt.close(fig)

    def test_plot_fft_linear_scale(self):
        """Test FFT with linear scale for narrow frequency range."""
        generator = IEEEPlotGenerator()
        # Narrow frequency range should keep linear scale
        frequencies = np.linspace(0, 100, 100)
        magnitude_db = np.random.randn(len(frequencies)) * 10 - 20

        fig = generator.plot_fft(frequencies, magnitude_db)

        ax = fig.axes[0]
        # Should use linear scale
        assert ax.get_xscale() == "linear"

        plt.close(fig)

    def test_plot_fft_empty_data(self):
        """Test FFT with empty arrays."""
        generator = IEEEPlotGenerator()
        frequencies = np.array([])
        magnitude_db = np.array([])

        fig = generator.plot_fft(frequencies, magnitude_db)

        assert fig is not None

        plt.close(fig)

    def test_plot_fft_dc_only(self):
        """Test FFT with only DC component."""
        generator = IEEEPlotGenerator()
        frequencies = np.array([0.0])
        magnitude_db = np.array([10.0])

        fig = generator.plot_fft(frequencies, magnitude_db, peak_markers=5)

        # Should not crash even with no valid peaks
        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotPSD:
    """Test Power Spectral Density plotting."""

    def test_plot_psd_basic(self):
        """Test basic PSD plotting."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        psd = np.random.rand(100) * 1e-6

        fig = generator.plot_psd(frequencies, psd)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_psd_custom_units(self):
        """Test PSD with custom units."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        psd = np.random.rand(100) * 1e-6

        fig = generator.plot_psd(frequencies, psd, units="A²/Hz")

        ax = fig.axes[0]
        assert "A²/Hz" in ax.get_ylabel()

        plt.close(fig)

    def test_plot_psd_log_scale(self):
        """Test PSD with log scale for wide frequency range."""
        generator = IEEEPlotGenerator()
        frequencies = np.logspace(0, 3, 100)
        psd = np.random.rand(100) * 1e-6

        fig = generator.plot_psd(frequencies, psd)

        ax = fig.axes[0]
        assert ax.get_xscale() == "log"

        plt.close(fig)

    def test_plot_psd_zero_values(self):
        """Test PSD with zero values (epsilon handling)."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        psd = np.zeros(100)  # All zeros

        fig = generator.plot_psd(frequencies, psd)

        # Should not crash due to log(0)
        assert fig is not None

        plt.close(fig)

    def test_plot_psd_empty_data(self):
        """Test PSD with empty arrays."""
        generator = IEEEPlotGenerator()
        frequencies = np.array([])
        psd = np.array([])

        fig = generator.plot_psd(frequencies, psd)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotSpectrogram:
    """Test spectrogram plotting."""

    def test_plot_spectrogram_basic(self):
        """Test basic spectrogram plotting."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        frequencies = np.linspace(0, 500, 50)
        spectrogram = np.random.rand(50, 100) * 1e-6

        fig = generator.plot_spectrogram(time, frequencies, spectrogram)

        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar

        plt.close(fig)

    def test_plot_spectrogram_custom_title(self):
        """Test spectrogram with custom title."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        frequencies = np.linspace(0, 500, 50)
        spectrogram = np.random.rand(50, 100) * 1e-6

        fig = generator.plot_spectrogram(time, frequencies, spectrogram, title="Custom Spec")

        ax = fig.axes[0]
        assert ax.get_title() == "Custom Spec"

        plt.close(fig)

    def test_plot_spectrogram_zero_values(self):
        """Test spectrogram with zero values (epsilon handling)."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        frequencies = np.linspace(0, 500, 50)
        spectrogram = np.zeros((50, 100))

        fig = generator.plot_spectrogram(time, frequencies, spectrogram)

        # Should not crash due to log(0)
        assert fig is not None

        plt.close(fig)

    def test_plot_spectrogram_single_time(self):
        """Test spectrogram with single time point."""
        generator = IEEEPlotGenerator()
        time = np.array([0.5])
        frequencies = np.linspace(0, 500, 50)
        spectrogram = np.random.rand(50, 1) * 1e-6

        fig = generator.plot_spectrogram(time, frequencies, spectrogram)

        assert fig is not None

        plt.close(fig)

    def test_plot_spectrogram_single_frequency(self):
        """Test spectrogram with single frequency bin."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        frequencies = np.array([100.0])
        spectrogram = np.random.rand(1, 100) * 1e-6

        fig = generator.plot_spectrogram(time, frequencies, spectrogram)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotEyeDiagram:
    """Test eye diagram plotting."""

    def test_plot_eye_diagram_basic(self):
        """Test basic eye diagram plotting."""
        generator = IEEEPlotGenerator()
        # Generate digital-like signal
        signal = np.tile(np.array([0, 0, 1, 1, 0, 1, 1, 0]), 125)  # 1000 samples
        samples_per_symbol = 10

        fig = generator.plot_eye_diagram(signal, samples_per_symbol)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_eye_diagram_custom_traces(self):
        """Test eye diagram with custom number of traces."""
        generator = IEEEPlotGenerator()
        signal = np.tile(np.array([0, 1, 1, 0]), 250)
        samples_per_symbol = 10

        fig = generator.plot_eye_diagram(signal, samples_per_symbol, num_traces=50)

        assert fig is not None

        plt.close(fig)

    def test_plot_eye_diagram_few_symbols(self):
        """Test eye diagram with very few symbols."""
        generator = IEEEPlotGenerator()
        signal = np.array([0, 1, 1, 0] * 5)  # Only 20 samples
        samples_per_symbol = 4

        fig = generator.plot_eye_diagram(signal, samples_per_symbol, num_traces=10)

        # Should limit traces to available symbols
        assert fig is not None

        plt.close(fig)

    def test_plot_eye_diagram_single_symbol(self):
        """Test eye diagram with single symbol."""
        generator = IEEEPlotGenerator()
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        samples_per_symbol = 8

        fig = generator.plot_eye_diagram(signal, samples_per_symbol, num_traces=1)

        assert fig is not None

        plt.close(fig)

    def test_plot_eye_diagram_short_signal(self):
        """Test eye diagram with very short signal."""
        generator = IEEEPlotGenerator()
        signal = np.array([0.0, 1.0, 0.0, 1.0])
        samples_per_symbol = 2

        fig = generator.plot_eye_diagram(signal, samples_per_symbol)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotHistogram:
    """Test histogram plotting."""

    def test_plot_histogram_basic(self):
        """Test basic histogram plotting."""
        generator = IEEEPlotGenerator()
        data = np.random.randn(1000)

        fig = generator.plot_histogram(data)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_histogram_custom_bins(self):
        """Test histogram with custom bin count."""
        generator = IEEEPlotGenerator()
        data = np.random.randn(1000)

        fig = generator.plot_histogram(data, bins=100)

        assert fig is not None

        plt.close(fig)

    def test_plot_histogram_custom_labels(self):
        """Test histogram with custom labels."""
        generator = IEEEPlotGenerator()
        data = np.random.randn(1000)

        fig = generator.plot_histogram(data, title="Custom Histogram", xlabel="Custom Units")

        ax = fig.axes[0]
        assert ax.get_title() == "Custom Histogram"
        assert ax.get_xlabel() == "Custom Units"

        plt.close(fig)

    def test_plot_histogram_gaussian_fit(self):
        """Test histogram with Gaussian fit overlay."""
        generator = IEEEPlotGenerator()
        # Use normal distribution for good fit
        data = np.random.randn(1000)

        fig = generator.plot_histogram(data)

        # Should have histogram and Gaussian fit line
        assert fig is not None
        # Check legend exists (for Gaussian fit)
        ax = fig.axes[0]
        assert ax.get_legend() is not None

        plt.close(fig)

    def test_plot_histogram_small_dataset(self):
        """Test histogram with small dataset."""
        generator = IEEEPlotGenerator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        fig = generator.plot_histogram(data, bins=5)

        assert fig is not None

        plt.close(fig)

    def test_plot_histogram_single_value(self):
        """Test histogram with constant data."""
        generator = IEEEPlotGenerator()
        data = np.ones(100)

        # RuntimeWarning for divide by zero or invalid value in Gaussian fit is expected
        with pytest.warns(RuntimeWarning):
            fig = generator.plot_histogram(data)

        assert fig is not None

        plt.close(fig)

    def test_plot_histogram_empty_data(self):
        """Test histogram with empty array."""
        generator = IEEEPlotGenerator()
        data = np.array([])

        # Empty data triggers warnings for mean/std/division calculations
        with pytest.warns(RuntimeWarning):
            fig = generator.plot_histogram(data)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotJitter:
    """Test jitter analysis plotting."""

    def test_plot_jitter_basic(self):
        """Test basic jitter plotting."""
        generator = IEEEPlotGenerator()
        # Time intervals in seconds
        time_intervals = np.random.randn(100) * 1e-9 + 1e-6  # ~1 µs with jitter

        fig = generator.plot_jitter(time_intervals)

        assert fig is not None
        assert len(fig.axes) == 2  # Time series + histogram

        plt.close(fig)

    def test_plot_jitter_custom_title(self):
        """Test jitter with custom title."""
        generator = IEEEPlotGenerator()
        time_intervals = np.random.randn(100) * 1e-9 + 1e-6

        fig = generator.plot_jitter(time_intervals, title="Custom Jitter Analysis")

        # Check that title is used in the plot
        assert fig is not None

        plt.close(fig)

    def test_plot_jitter_statistics(self):
        """Test jitter statistics are computed and displayed."""
        generator = IEEEPlotGenerator()
        time_intervals = np.random.randn(100) * 1e-9 + 1e-6

        fig = generator.plot_jitter(time_intervals)

        # Should have statistics text box in histogram
        assert fig is not None

        plt.close(fig)

    def test_plot_jitter_small_dataset(self):
        """Test jitter with small dataset."""
        generator = IEEEPlotGenerator()
        time_intervals = np.array([1e-6, 1.1e-6, 0.9e-6, 1.05e-6, 0.95e-6])

        fig = generator.plot_jitter(time_intervals)

        assert fig is not None

        plt.close(fig)

    def test_plot_jitter_single_value(self):
        """Test jitter with single interval."""
        generator = IEEEPlotGenerator()
        time_intervals = np.array([1e-6])

        fig = generator.plot_jitter(time_intervals)

        assert fig is not None

        plt.close(fig)

    def test_plot_jitter_zero_jitter(self):
        """Test jitter with constant intervals (zero jitter)."""
        generator = IEEEPlotGenerator()
        time_intervals = np.ones(100) * 1e-6

        fig = generator.plot_jitter(time_intervals)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestPlotPower:
    """Test power waveform plotting."""

    def test_plot_power_basic(self):
        """Test basic power plotting."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 1000)
        voltage = np.sin(2 * np.pi * 60 * time) * 120
        current = np.sin(2 * np.pi * 60 * time) * 10

        fig = generator.plot_power(time, voltage, current)

        assert fig is not None
        assert len(fig.axes) == 3  # Voltage, current, power subplots

        plt.close(fig)

    def test_plot_power_custom_title(self):
        """Test power plot with custom title."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        voltage = np.sin(2 * np.pi * 60 * time) * 120
        current = np.sin(2 * np.pi * 60 * time) * 10

        fig = generator.plot_power(time, voltage, current, title="AC Power Analysis")

        # Main title should be set (check using get_suptitle)
        suptitle = fig.get_suptitle()
        assert suptitle is not None
        assert suptitle == "AC Power Analysis"

        plt.close(fig)

    def test_plot_power_dc_signals(self):
        """Test power plot with DC signals."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        voltage = np.ones(100) * 5.0  # 5V DC
        current = np.ones(100) * 2.0  # 2A DC

        fig = generator.plot_power(time, voltage, current)

        assert fig is not None

        plt.close(fig)

    def test_plot_power_zero_current(self):
        """Test power plot with zero current."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        voltage = np.sin(2 * np.pi * 60 * time) * 120
        current = np.zeros(100)

        fig = generator.plot_power(time, voltage, current)

        assert fig is not None

        plt.close(fig)

    def test_plot_power_negative_values(self):
        """Test power plot with negative values (bidirectional power)."""
        generator = IEEEPlotGenerator()
        time = np.linspace(0, 1, 100)
        voltage = np.sin(2 * np.pi * 60 * time) * 120
        # Phase-shifted current for reactive power
        current = np.sin(2 * np.pi * 60 * time + np.pi / 2) * 10

        fig = generator.plot_power(time, voltage, current)

        # Should have zero line for reference
        assert fig is not None

        plt.close(fig)

    def test_plot_power_single_point(self):
        """Test power plot with single data point."""
        generator = IEEEPlotGenerator()
        time = np.array([0.5])
        voltage = np.array([5.0])
        current = np.array([2.0])

        fig = generator.plot_power(time, voltage, current)

        assert fig is not None

        plt.close(fig)


@skip_if_no_matplotlib
class TestFigureToBase64:
    """Test figure to base64 conversion."""

    def test_figure_to_base64_png(self):
        """Test conversion to PNG base64."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])

        img_str = IEEEPlotGenerator.figure_to_base64(fig, format="png")

        # Check format
        assert img_str.startswith("data:image/png;base64,")
        # Check it's valid base64
        base64_data = img_str.split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

        # Note: figure is closed by figure_to_base64

    def test_figure_to_base64_jpg(self):
        """Test conversion to JPG base64."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])

        img_str = IEEEPlotGenerator.figure_to_base64(fig, format="jpg")

        assert img_str.startswith("data:image/jpg;base64,")
        base64_data = img_str.split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_figure_to_base64_svg(self):
        """Test conversion to SVG base64."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])

        img_str = IEEEPlotGenerator.figure_to_base64(fig, format="svg")

        assert img_str.startswith("data:image/svg;base64,")
        base64_data = img_str.split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_figure_to_base64_closes_figure(self):
        """Test that figure is closed after conversion."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])

        # Get figure number
        fig_num = fig.number

        IEEEPlotGenerator.figure_to_base64(fig)

        # Figure should be closed
        assert fig_num not in plt.get_fignums()

    def test_figure_to_base64_complex_plot(self):
        """Test conversion of complex plot with multiple elements."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)
        markers = {"Peak": 0.25}

        fig = generator.plot_waveform(t, signal, markers=markers)
        img_str = IEEEPlotGenerator.figure_to_base64(fig)

        assert img_str.startswith("data:image/png;base64,")
        assert len(img_str) > 100  # Should have substantial data


@skip_if_no_matplotlib
class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions across all plot types."""

    def test_large_dataset_waveform(self):
        """Test waveform with large dataset."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100000)
        signal = np.sin(2 * np.pi * 10 * t)

        fig = generator.plot_waveform(t, signal)

        assert fig is not None
        plt.close(fig)

    def test_negative_values_fft(self):
        """Test FFT with negative magnitude values."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        # Negative dB values are common
        magnitude_db = np.random.randn(len(frequencies)) * 20 - 60

        fig = generator.plot_fft(frequencies, magnitude_db)

        assert fig is not None
        plt.close(fig)

    def test_very_small_psd_values(self):
        """Test PSD with very small values."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        psd = np.random.rand(100) * 1e-15  # Very small values

        fig = generator.plot_psd(frequencies, psd)

        assert fig is not None
        plt.close(fig)

    def test_nan_values_handling(self):
        """Test that NaN values don't crash plotting."""
        generator = IEEEPlotGenerator()
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)
        signal[50] = np.nan  # Insert NaN

        fig = generator.plot_waveform(t, signal)

        assert fig is not None
        plt.close(fig)

    def test_inf_values_handling(self):
        """Test that inf values don't crash plotting."""
        generator = IEEEPlotGenerator()
        frequencies = np.linspace(0, 500, 100)
        magnitude_db = np.random.randn(len(frequencies)) * 10 - 20
        magnitude_db[50] = np.inf  # Insert inf

        fig = generator.plot_fft(frequencies, magnitude_db)

        assert fig is not None
        plt.close(fig)


class TestModuleConstants:
    """Test module-level constants and imports."""

    def test_has_matplotlib_constant(self):
        """Test that HAS_MATPLOTLIB constant is defined."""
        assert isinstance(MODULE_HAS_MATPLOTLIB, bool)
        # Should match our local check
        assert MODULE_HAS_MATPLOTLIB == HAS_MATPLOTLIB

    def test_ieee_colors_immutable(self):
        """Test that IEEE_COLORS is defined and accessible."""
        # Should be a dict
        assert isinstance(IEEE_COLORS, dict)
        # Should have expected keys
        assert "primary" in IEEE_COLORS
        assert "secondary" in IEEE_COLORS


def test_matplotlib_not_installed_behavior():
    """Test behavior when matplotlib is not installed."""
    if not HAS_MATPLOTLIB:
        # PlotStyler.apply_ieee_style should return early
        PlotStyler.apply_ieee_style(None, "X", "Y", "Title")  # Should not crash

        # IEEEPlotGenerator should raise ImportError
        with pytest.raises(ImportError, match="matplotlib is required"):
            IEEEPlotGenerator()
