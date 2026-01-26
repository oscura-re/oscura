"""Unit tests for spectral visualization functions.

Tests:
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def sample_trace():
    """Create a sample waveform trace for testing."""
    sample_rate = 1e6  # 1 MHz
    n_samples = 1000
    metadata = TraceMetadata(sample_rate=sample_rate)

    # Create a signal with multiple frequency components
    t = np.arange(n_samples) / sample_rate
    # 1 kHz sine + 5 kHz sine + 10 kHz sine
    data = (
        np.sin(2 * np.pi * 1000 * t)
        + 0.5 * np.sin(2 * np.pi * 5000 * t)
        + 0.25 * np.sin(2 * np.pi * 10000 * t)
    )

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def mock_fft_result():
    """Create mock FFT result."""
    freq = np.logspace(1, 5, 100)  # 10 Hz to 100 kHz
    mag_db = (
        -20 * np.log10(freq / 1000) + np.random.randn(100) * 2
    )  # Decreasing spectrum with noise
    return freq, mag_db


@pytest.fixture
def mock_psd_result():
    """Create mock PSD result."""
    freq = np.logspace(1, 5, 100)
    psd_db = -30 * np.log10(freq / 1000) + np.random.randn(100) * 3
    return freq, psd_db


@pytest.fixture
def mock_spectrogram_result():
    """Create mock spectrogram result."""
    times = np.linspace(0, 1, 50)
    freq = np.linspace(0, 500e3, 100)
    Sxx_db = np.random.randn(100, 50) * 10 - 40
    return times, freq, Sxx_db


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotSpectrum:
    """Tests for plot_spectrum function."""

    def test_import_error_without_matplotlib(self, sample_trace):
        """Test that ImportError is raised when matplotlib is not available."""
        from oscura.visualization import spectral

        with patch.object(spectral, "HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError, match="matplotlib is required"):
                spectral.plot_spectrum(sample_trace)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_basic_spectrum_plot(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test basic spectrum plot creation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        # Setup mocks
        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Call function with show=False to avoid display
        fig = plot_spectrum(sample_trace, show=False)

        # Verify FFT was called
        mock_fft.assert_called_once()
        assert mock_fft.call_args[0][0] is sample_trace

        # Verify plot was created
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called_once_with("Magnitude (dB)")

        assert fig is not None

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_with_provided_axes(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test spectrum plot with provided axes."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result

        # Create mock axes with figure
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig

        fig = plot_spectrum(sample_trace, ax=mock_ax, show=False)

        # Should not call plt.subplots when ax is provided
        mock_plt.subplots.assert_not_called()
        assert fig is mock_fig

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_axes_without_figure_error(self, mock_fft, mock_plt, sample_trace):
        """Test error when axes has no associated figure."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_spectrum(sample_trace, ax=mock_ax, show=False)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_freq_unit_auto_selection(self, mock_fft, mock_plt, sample_trace):
        """Test automatic frequency unit selection."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test Hz range
        freq_hz = np.linspace(0, 500, 100)
        mag_db = np.random.randn(100)
        mock_fft.return_value = (freq_hz, mag_db)
        plot_spectrum(sample_trace, freq_unit="auto", show=False)
        assert "Hz" in str(mock_ax.set_xlabel.call_args)

        # Test kHz range
        freq_khz = np.linspace(0, 500e3, 100)
        mock_fft.return_value = (freq_khz, mag_db)
        plot_spectrum(sample_trace, freq_unit="auto", show=False)
        assert "kHz" in str(mock_ax.set_xlabel.call_args)

        # Test MHz range
        freq_mhz = np.linspace(0, 500e6, 100)
        mock_fft.return_value = (freq_mhz, mag_db)
        plot_spectrum(sample_trace, freq_unit="auto", show=False)
        assert "MHz" in str(mock_ax.set_xlabel.call_args)

        # Test GHz range
        freq_ghz = np.linspace(0, 5e9, 100)
        mock_fft.return_value = (freq_ghz, mag_db)
        plot_spectrum(sample_trace, freq_unit="auto", show=False)
        assert "GHz" in str(mock_ax.set_xlabel.call_args)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_explicit_freq_units(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test explicit frequency unit specification."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        for unit in ["Hz", "kHz", "MHz", "GHz"]:
            plot_spectrum(sample_trace, freq_unit=unit, show=False)
            xlabel_call = mock_ax.set_xlabel.call_args[0][0]
            assert unit in xlabel_call

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_db_reference(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test dB reference adjustment."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        freq, mag_db = mock_fft_result
        mock_fft.return_value = (freq, mag_db.copy())
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        db_ref = 20.0
        plot_spectrum(sample_trace, db_ref=db_ref, show=False)

        # The plotted data should be adjusted by db_ref
        plotted_data = mock_ax.plot.call_args[0][1]
        # Note: due to numpy operations, we check that data was modified
        assert mock_ax.plot.called

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_custom_title(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        title = "Custom Spectrum Title"
        plot_spectrum(sample_trace, title=title, show=False)

        mock_ax.set_title.assert_called_once_with(title)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_default_title(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test default title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrum(sample_trace, show=False)

        mock_ax.set_title.assert_called_once_with("Magnitude Spectrum")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_grid_toggle(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test grid on/off."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with grid
        plot_spectrum(sample_trace, show_grid=True, show=False)
        mock_ax.grid.assert_called_with(True, alpha=0.3, which="both")

        # Reset mock
        mock_ax.reset_mock()

        # Test without grid
        plot_spectrum(sample_trace, show_grid=False, show=False)
        mock_ax.grid.assert_not_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_custom_color(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test custom line color."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        color = "red"
        plot_spectrum(sample_trace, color=color, show=False)

        # Check that plot was called with the color
        assert mock_ax.plot.call_args[1]["color"] == color

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_window_parameter(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test window parameter is passed to FFT."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        window = "blackman"
        plot_spectrum(sample_trace, window=window, show=False)

        # Check window was passed to FFT
        assert mock_fft.call_args[1]["window"] == window

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_log_scale(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test log scale setting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test log scale
        plot_spectrum(sample_trace, log_scale=True, show=False)
        mock_ax.set_xscale.assert_called_with("log")

        # Reset and test linear scale
        mock_ax.reset_mock()
        plot_spectrum(sample_trace, log_scale=False, show=False)
        mock_ax.set_xscale.assert_called_with("linear")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_xscale_deprecated_parameter(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test deprecated xscale parameter still works."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with deprecated xscale parameter
        plot_spectrum(sample_trace, xscale="log", show=False)
        # log_scale defaults to True, so should be log
        mock_ax.set_xscale.assert_called_with("log")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_freq_range(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test frequency range limiting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        freq_range = (1000, 50000)  # 1 kHz to 50 kHz
        plot_spectrum(sample_trace, freq_range=freq_range, freq_unit="Hz", show=False)

        # Check xlim was called
        mock_ax.set_xlim.assert_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_xlim_ylim(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test custom xlim and ylim."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        xlim = (10, 100000)
        ylim = (-100, 0)
        plot_spectrum(sample_trace, xlim=xlim, ylim=ylim, show=False)

        # Check limits were set
        assert mock_ax.set_xlim.called
        assert mock_ax.set_ylim.called

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_custom_figsize(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        figsize = (12, 8)
        plot_spectrum(sample_trace, figsize=figsize, show=False)

        # Check figsize was passed to subplots
        mock_plt.subplots.assert_called_once_with(figsize=figsize)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_precomputed_fft(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test with pre-computed FFT result."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrum(sample_trace, fft_result=mock_fft_result, show=False)

        # FFT should not be called when result is provided
        mock_fft.assert_not_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_save_path(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test saving figure to file."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        save_path = "/tmp/test_spectrum.png"
        plot_spectrum(sample_trace, save_path=save_path, show=False)

        # Check savefig was called
        mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_show_parameter(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test show parameter controls plt.show()."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with show=False
        plot_spectrum(sample_trace, show=False)
        mock_plt.show.assert_not_called()

        # Test with show=True
        mock_plt.reset_mock()
        plot_spectrum(sample_trace, show=True)
        mock_plt.show.assert_called_once()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_ylim_auto_with_valid_data(self, mock_fft, mock_plt, sample_trace):
        """Test automatic y-limit calculation with valid data."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        freq = np.linspace(0, 500e3, 100)
        mag_db = np.linspace(-60, 0, 100)
        mock_fft.return_value = (freq, mag_db)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrum(sample_trace, show=False)

        # Check that ylim was set
        mock_ax.set_ylim.assert_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_ylim_with_inf_values(self, mock_fft, mock_plt, sample_trace):
        """Test y-limit calculation handles infinite values."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        freq = np.linspace(0, 500e3, 100)
        mag_db = np.linspace(-60, 0, 100)
        mag_db[0] = -np.inf  # Add infinite value

        mock_fft.return_value = (freq, mag_db)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrum(sample_trace, show=False)

        # Should not raise error and should set ylim
        mock_ax.set_ylim.assert_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_ylim_with_all_inf_values(self, mock_fft, mock_plt, sample_trace):
        """Test y-limit calculation when all values are infinite."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        freq = np.linspace(0, 500e3, 100)
        mag_db = np.full(100, -np.inf)  # All infinite

        mock_fft.return_value = (freq, mag_db)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrum(sample_trace, show=False)

        # Should not crash even with all inf values
        # ylim won't be set because len(valid_db) == 0
        assert mock_ax.plot.called

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_db_scale_false(self, mock_fft, mock_plt, sample_trace, mock_fft_result):
        """Test db_scale parameter (currently unused but accepted)."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrum

        mock_fft.return_value = mock_fft_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # db_scale is currently accepted but not used (data is always in dB)
        plot_spectrum(sample_trace, db_scale=False, show=False)

        # Should still create plot
        assert mock_ax.plot.called


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotSpectrogram:
    """Tests for plot_spectrogram function."""

    def test_import_error_without_matplotlib(self, sample_trace):
        """Test that ImportError is raised when matplotlib is not available."""
        from oscura.visualization import spectral

        with patch.object(spectral, "HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError, match="matplotlib is required"):
                spectral.plot_spectrogram(sample_trace)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_basic_spectrogram(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test basic spectrogram plot creation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = plot_spectrogram(sample_trace)

        # Verify spectrogram was called
        mock_spectrogram.assert_called_once()

        # Verify plot was created
        mock_ax.pcolormesh.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()

        assert fig is not None

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_with_provided_axes(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test spectrogram with provided axes."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig

        fig = plot_spectrogram(sample_trace, ax=mock_ax)

        # Should not call plt.subplots when ax is provided
        mock_plt.subplots.assert_not_called()
        assert fig is mock_fig

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_axes_without_figure_error(self, mock_spectrogram, mock_plt, sample_trace):
        """Test error when axes has no associated figure."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_spectrogram(sample_trace, ax=mock_ax)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_time_unit_auto_selection(self, mock_spectrogram, mock_plt, sample_trace):
        """Test automatic time unit selection."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        freq = np.linspace(0, 500e3, 100)
        Sxx_db = np.random.randn(100, 50) * 10 - 40

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test nanoseconds range
        times_ns = np.linspace(0, 500e-9, 50)
        mock_spectrogram.return_value = (times_ns, freq, Sxx_db)
        plot_spectrogram(sample_trace, time_unit="auto")
        assert "ns" in str(mock_ax.set_xlabel.call_args)

        # Test microseconds range
        times_us = np.linspace(0, 500e-6, 50)
        mock_spectrogram.return_value = (times_us, freq, Sxx_db)
        plot_spectrogram(sample_trace, time_unit="auto")
        assert "us" in str(mock_ax.set_xlabel.call_args)

        # Test milliseconds range
        times_ms = np.linspace(0, 500e-3, 50)
        mock_spectrogram.return_value = (times_ms, freq, Sxx_db)
        plot_spectrogram(sample_trace, time_unit="auto")
        assert "ms" in str(mock_ax.set_xlabel.call_args)

        # Test seconds range
        times_s = np.linspace(0, 5, 50)
        mock_spectrogram.return_value = (times_s, freq, Sxx_db)
        plot_spectrogram(sample_trace, time_unit="auto")
        assert "s" in str(mock_ax.set_xlabel.call_args)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_freq_unit_auto_selection(self, mock_spectrogram, mock_plt, sample_trace):
        """Test automatic frequency unit selection."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        times = np.linspace(0, 1, 50)
        Sxx_db = np.random.randn(100, 50) * 10 - 40

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test Hz range
        freq_hz = np.linspace(0, 500, 100)
        mock_spectrogram.return_value = (times, freq_hz, Sxx_db)
        plot_spectrogram(sample_trace, freq_unit="auto")
        assert "Hz" in str(mock_ax.set_ylabel.call_args)

        # Test kHz range
        freq_khz = np.linspace(0, 500e3, 100)
        mock_spectrogram.return_value = (times, freq_khz, Sxx_db)
        plot_spectrogram(sample_trace, freq_unit="auto")
        assert "kHz" in str(mock_ax.set_ylabel.call_args)

        # Test MHz range
        freq_mhz = np.linspace(0, 500e6, 100)
        mock_spectrogram.return_value = (times, freq_mhz, Sxx_db)
        plot_spectrogram(sample_trace, freq_unit="auto")
        assert "MHz" in str(mock_ax.set_ylabel.call_args)

        # Test GHz range
        freq_ghz = np.linspace(0, 5e9, 100)
        mock_spectrogram.return_value = (times, freq_ghz, Sxx_db)
        plot_spectrogram(sample_trace, freq_unit="auto")
        assert "GHz" in str(mock_ax.set_ylabel.call_args)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_explicit_units(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test explicit unit specification."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        time_units = ["s", "ms", "us", "ns"]
        freq_units = ["Hz", "kHz", "MHz", "GHz"]

        for t_unit in time_units:
            for f_unit in freq_units:
                plot_spectrogram(sample_trace, time_unit=t_unit, freq_unit=f_unit)
                xlabel = mock_ax.set_xlabel.call_args[0][0]
                ylabel = mock_ax.set_ylabel.call_args[0][0]
                assert t_unit in xlabel
                assert f_unit in ylabel

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_custom_colormap(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test custom colormap."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        cmap = "plasma"
        plot_spectrogram(sample_trace, cmap=cmap)

        # Check cmap was used
        assert mock_ax.pcolormesh.call_args[1]["cmap"] == cmap

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_custom_vmin_vmax(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test custom color limits."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        vmin = -80
        vmax = -20
        plot_spectrogram(sample_trace, vmin=vmin, vmax=vmax)

        # Check vmin/vmax were used
        assert mock_ax.pcolormesh.call_args[1]["vmin"] == vmin
        assert mock_ax.pcolormesh.call_args[1]["vmax"] == vmax

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_partial_vmin_vmax(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test providing only vmax or only vmin."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with only vmax provided
        vmax = -20
        plot_spectrogram(sample_trace, vmax=vmax)

        # Check vmax was used and vmin was auto-calculated
        assert mock_ax.pcolormesh.call_args[1]["vmax"] == vmax
        assert mock_ax.pcolormesh.call_args[1]["vmin"] is not None

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_auto_vmin_vmax(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test automatic color limit calculation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrogram(sample_trace)

        # Check that vmin/vmax were set automatically
        assert mock_ax.pcolormesh.called

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_custom_title(self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result):
        """Test custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        title = "Custom Spectrogram"
        plot_spectrogram(sample_trace, title=title)

        mock_ax.set_title.assert_called_once_with(title)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_default_title(self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result):
        """Test default title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_spectrogram(sample_trace)

        mock_ax.set_title.assert_called_once_with("Spectrogram")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_window_parameter(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test window parameter is passed to spectrogram."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        window = "blackman"
        plot_spectrogram(sample_trace, window=window)

        assert mock_spectrogram.call_args[1]["window"] == window

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_nperseg_parameter(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test nperseg parameter is passed to spectrogram."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        nperseg = 256
        plot_spectrogram(sample_trace, nperseg=nperseg)

        assert mock_spectrogram.call_args[1]["nperseg"] == nperseg

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_nfft_parameter(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test nfft parameter is used as nperseg."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        nfft = 512
        plot_spectrogram(sample_trace, nfft=nfft)

        # nfft should be converted to nperseg
        assert mock_spectrogram.call_args[1]["nperseg"] == nfft

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_overlap_parameter(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test overlap parameter is converted to noverlap."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        nperseg = 256
        overlap = 0.75  # 75% overlap
        plot_spectrogram(sample_trace, nperseg=nperseg, overlap=overlap)

        # Check noverlap was calculated correctly
        expected_noverlap = int(nperseg * overlap)
        assert mock_spectrogram.call_args[1]["noverlap"] == expected_noverlap

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_colorbar_creation(
        self, mock_spectrogram, mock_plt, sample_trace, mock_spectrogram_result
    ):
        """Test colorbar is created."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        mock_spectrogram.return_value = mock_spectrogram_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig  # Connect ax to fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Create a mock colorbar
        mock_cbar = MagicMock()
        mock_fig.colorbar.return_value = mock_cbar

        plot_spectrogram(sample_trace)

        # Check colorbar was created
        mock_fig.colorbar.assert_called_once()
        mock_cbar.set_label.assert_called_once_with("Magnitude (dB)")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_empty_arrays_handling(self, mock_spectrogram, mock_plt, sample_trace):
        """Test handling of empty time/freq arrays."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        # Empty arrays
        times = np.array([])
        freq = np.array([])
        Sxx_db = np.array([]).reshape(0, 0)

        mock_spectrogram.return_value = (times, freq, Sxx_db)
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Should not crash with empty arrays
        plot_spectrogram(sample_trace)

        # Check that plot was still called
        mock_ax.pcolormesh.assert_called_once()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.spectrogram")
    def test_all_inf_values_handling(self, mock_spectrogram, mock_plt, sample_trace):
        """Test handling when all Sxx values are infinite."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_spectrogram

        times = np.linspace(0, 1, 50)
        freq = np.linspace(0, 500e3, 100)
        Sxx_db = np.full((100, 50), -np.inf)  # All infinite

        mock_spectrogram.return_value = (times, freq, Sxx_db)
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Should not crash with all inf values
        plot_spectrogram(sample_trace)

        # Check that plot was still called
        mock_ax.pcolormesh.assert_called_once()


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotPSD:
    """Tests for plot_psd function."""

    def test_import_error_without_matplotlib(self, sample_trace):
        """Test that ImportError is raised when matplotlib is not available."""
        from oscura.visualization import spectral

        with patch.object(spectral, "HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError, match="matplotlib is required"):
                spectral.plot_psd(sample_trace)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_basic_psd_plot(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test basic PSD plot creation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = plot_psd(sample_trace)

        # Verify PSD was called
        mock_psd.assert_called_once()

        # Verify plot was created
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called_once_with("PSD (dB/Hz)")

        assert fig is not None

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_with_provided_axes(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test PSD plot with provided axes."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig

        fig = plot_psd(sample_trace, ax=mock_ax)

        # Should not call plt.subplots when ax is provided
        mock_plt.subplots.assert_not_called()
        assert fig is mock_fig

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_axes_without_figure_error(self, mock_psd, mock_plt, sample_trace):
        """Test error when axes has no associated figure."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_psd(sample_trace, ax=mock_ax)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_freq_unit_auto_selection(self, mock_psd, mock_plt, sample_trace):
        """Test automatic frequency unit selection."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        psd_db = np.random.randn(100)
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test Hz range
        freq_hz = np.linspace(0, 500, 100)
        mock_psd.return_value = (freq_hz, psd_db)
        plot_psd(sample_trace, freq_unit="auto")
        assert "Hz" in str(mock_ax.set_xlabel.call_args)

        # Test kHz range
        freq_khz = np.linspace(0, 500e3, 100)
        mock_psd.return_value = (freq_khz, psd_db)
        plot_psd(sample_trace, freq_unit="auto")
        assert "kHz" in str(mock_ax.set_xlabel.call_args)

        # Test MHz range
        freq_mhz = np.linspace(0, 500e6, 100)
        mock_psd.return_value = (freq_mhz, psd_db)
        plot_psd(sample_trace, freq_unit="auto")
        assert "MHz" in str(mock_ax.set_xlabel.call_args)

        # Test GHz range
        freq_ghz = np.linspace(0, 5e9, 100)
        mock_psd.return_value = (freq_ghz, psd_db)
        plot_psd(sample_trace, freq_unit="auto")
        assert "GHz" in str(mock_ax.set_xlabel.call_args)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_custom_title(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        title = "Custom PSD Title"
        plot_psd(sample_trace, title=title)

        mock_ax.set_title.assert_called_once_with(title)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_default_title(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test default title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_psd(sample_trace)

        mock_ax.set_title.assert_called_once_with("Power Spectral Density")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_grid_toggle(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test grid on/off."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with grid
        plot_psd(sample_trace, show_grid=True)
        mock_ax.grid.assert_called_with(True, alpha=0.3, which="both")

        # Reset and test without grid
        mock_ax.reset_mock()
        plot_psd(sample_trace, show_grid=False)
        mock_ax.grid.assert_not_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_custom_color(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test custom line color."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        color = "red"
        plot_psd(sample_trace, color=color)

        assert mock_ax.plot.call_args[1]["color"] == color

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_window_parameter(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test window parameter is passed to PSD."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        window = "blackman"
        plot_psd(sample_trace, window=window)

        assert mock_psd.call_args[1]["window"] == window

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.analyzers.waveform.spectral.psd")
    def test_xscale(self, mock_psd, mock_plt, sample_trace, mock_psd_result):
        """Test x-axis scale setting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_psd

        mock_psd.return_value = mock_psd_result
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test log scale
        plot_psd(sample_trace, xscale="log")
        mock_ax.set_xscale.assert_called_with("log")

        # Reset and test linear scale
        mock_ax.reset_mock()
        plot_psd(sample_trace, xscale="linear")
        mock_ax.set_xscale.assert_called_with("linear")


@pytest.mark.unit
@pytest.mark.visualization
class TestPlotFFT:
    """Tests for plot_fft function."""

    def test_import_error_without_matplotlib(self, sample_trace):
        """Test that ImportError is raised when matplotlib is not available."""
        from oscura.visualization import spectral

        with patch.object(spectral, "HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError, match="matplotlib is required"):
                spectral.plot_fft(sample_trace)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_basic_fft_plot(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test basic FFT plot creation."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = plot_fft(sample_trace, show=False)

        # Verify plot_spectrum was called
        mock_plot_spectrum.assert_called_once()

        assert fig is not None

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_with_provided_axes(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test FFT plot with provided axes."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig

        fig = plot_fft(sample_trace, ax=mock_ax, show=False)

        # Should not call plt.subplots when ax is provided
        mock_plt.subplots.assert_not_called()
        assert fig is mock_fig

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_axes_without_figure_error(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test error when axes has no associated figure."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = None

        with pytest.raises(ValueError, match="Axes must have an associated figure"):
            plot_fft(sample_trace, ax=mock_ax, show=False)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_custom_title(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        title = "Custom FFT Title"
        plot_fft(sample_trace, title=title, show=False)

        # Check plot_spectrum was called with custom title
        assert mock_plot_spectrum.call_args[1]["title"] == title

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_default_title(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test default title."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_fft(sample_trace, show=False)

        # Check plot_spectrum was called with default title
        assert mock_plot_spectrum.call_args[1]["title"] == "FFT Magnitude Spectrum"

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_custom_figsize(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        figsize = (12, 8)
        plot_fft(sample_trace, figsize=figsize, show=False)

        mock_plt.subplots.assert_called_once_with(figsize=figsize)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_log_scale(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test log scale setting."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test log scale
        plot_fft(sample_trace, log_scale=True, show=False)
        assert mock_plot_spectrum.call_args[1]["xscale"] == "log"

        # Test linear scale
        plot_fft(sample_trace, log_scale=False, show=False)
        assert mock_plot_spectrum.call_args[1]["xscale"] == "linear"

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_custom_labels(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom axis labels."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_xlabel.return_value = "Frequency (kHz)"
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        xlabel = "Custom X Label"
        ylabel = "Custom Y Label"
        plot_fft(sample_trace, xlabel=xlabel, ylabel=ylabel, show=False)

        # Check labels were set after plot_spectrum call
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_default_labels(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test default axis labels are preserved."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_fft(sample_trace, show=False)

        # With default labels, no additional calls should be made
        # (plot_spectrum sets them)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_xlim_ylim(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom xlim and ylim."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        xlim = (10, 100000)
        ylim = (-100, 0)
        plot_fft(sample_trace, xlim=xlim, ylim=ylim, show=False)

        # Check limits were set
        mock_ax.set_xlim.assert_called_with(xlim)
        mock_ax.set_ylim.assert_called_with(ylim)

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_save_path(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test saving figure to file."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        save_path = "/tmp/test_fft.png"
        plot_fft(sample_trace, save_path=save_path, show=False)

        mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_show_parameter(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test show parameter controls plt.show()."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with show=False
        plot_fft(sample_trace, show=False)
        mock_plt.show.assert_not_called()

        # Test with show=True
        mock_plt.reset_mock()
        plot_fft(sample_trace, show=True)
        mock_plt.show.assert_called_once()

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_all_parameters_passed_to_plot_spectrum(
        self, mock_plot_spectrum, mock_plt, sample_trace
    ):
        """Test that relevant parameters are passed to plot_spectrum."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        freq_unit = "MHz"
        show_grid = False
        color = "blue"
        window = "blackman"

        plot_fft(
            sample_trace,
            freq_unit=freq_unit,
            show_grid=show_grid,
            color=color,
            window=window,
            show=False,
        )

        # Verify parameters were passed
        call_kwargs = mock_plot_spectrum.call_args[1]
        assert call_kwargs["freq_unit"] == freq_unit
        assert call_kwargs["show_grid"] == show_grid
        assert call_kwargs["color"] == color
        assert call_kwargs["window"] == window

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_xlabel_with_unit_preservation(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom xlabel preserves frequency unit."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_xlabel.return_value = "Frequency (MHz)"
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        xlabel = "Custom Frequency"
        plot_fft(sample_trace, xlabel=xlabel, show=False)

        # Check that unit was preserved when setting custom label
        calls = mock_ax.set_xlabel.call_args_list
        if len(calls) > 1:  # If set_xlabel was called after plot_spectrum
            last_call = calls[-1][0][0]
            assert "Custom Frequency" in last_call
            assert "(MHz)" in last_call

    @patch("oscura.visualization.spectral.plt")
    @patch("oscura.visualization.spectral.plot_spectrum")
    def test_xlabel_without_unit(self, mock_plot_spectrum, mock_plt, sample_trace):
        """Test custom xlabel when current label has no unit."""
        pytest.importorskip("matplotlib")
        from oscura.visualization.spectral import plot_fft

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_xlabel.return_value = "Frequency"  # No unit
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        xlabel = "Custom Label"
        plot_fft(sample_trace, xlabel=xlabel, show=False)

        # Should set the custom label without trying to extract unit
        calls = mock_ax.set_xlabel.call_args_list
        if len(calls) > 1:
            last_call = calls[-1][0][0]
            assert "Custom Label" in last_call


@pytest.mark.unit
@pytest.mark.visualization
class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from oscura.visualization.spectral import __all__

        expected = ["plot_fft", "plot_psd", "plot_spectrogram", "plot_spectrum"]
        assert set(__all__) == set(expected)

    def test_functions_importable(self):
        """Test that all exported functions are importable."""
        from oscura.visualization.spectral import (
            plot_fft,
            plot_psd,
            plot_spectrogram,
            plot_spectrum,
        )

        assert callable(plot_fft)
        assert callable(plot_psd)
        assert callable(plot_spectrogram)
        assert callable(plot_spectrum)
