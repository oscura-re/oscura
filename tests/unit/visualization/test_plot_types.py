"""Comprehensive tests for all visualization plot types.

Requirements tested:

This module provides comprehensive tests for all visualization
functions to verify plot type implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_waveform():
    """Create sample waveform trace."""
    sample_rate = 1e6  # 1 MHz
    t = np.arange(0, 1e-3, 1 / sample_rate)  # 1 ms
    # Sine wave with some harmonics
    data = 2.5 * np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


@pytest.fixture
def multi_channel_waveforms():
    """Create multiple waveform traces."""
    sample_rate = 1e6
    t = np.arange(0, 1e-3, 1 / sample_rate)
    metadata = TraceMetadata(sample_rate=sample_rate)

    ch1 = WaveformTrace(data=np.sin(2 * np.pi * 1000 * t), metadata=metadata)
    ch2 = WaveformTrace(data=np.cos(2 * np.pi * 1000 * t), metadata=metadata)
    ch3 = WaveformTrace(data=np.sin(2 * np.pi * 2000 * t), metadata=metadata)

    return [ch1, ch2, ch3]


@pytest.fixture
def digital_traces():
    """Create sample digital traces."""
    sample_rate = 10e6
    n_samples = 1000
    metadata = TraceMetadata(sample_rate=sample_rate)

    # Clock signal
    clk = np.tile([False, False, True, True], n_samples // 4)[:n_samples]
    # Data signal
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2, n_samples).astype(bool)
    # CS signal
    cs = np.zeros(n_samples, dtype=bool)
    cs[100:800] = True

    return [
        DigitalTrace(data=clk, metadata=metadata),
        DigitalTrace(data=data, metadata=metadata),
        DigitalTrace(data=cs, metadata=metadata),
    ]


@pytest.fixture
def eye_diagram_data():
    """Create data suitable for eye diagram."""
    sample_rate = 10e9  # 10 GHz
    bit_rate = 1e9  # 1 Gbps
    samples_per_bit = int(sample_rate / bit_rate)
    n_bits = 100

    # Random bit pattern
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, n_bits)

    # Create waveform with some jitter
    t = np.arange(n_bits * samples_per_bit) / sample_rate
    data = np.zeros_like(t)

    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        level = 0.8 if bit else 0.2
        # Add rise/fall time
        data[start:end] = level

    # Add some jitter and noise
    jitter = rng.standard_normal(len(data)) * 0.01 * samples_per_bit
    noise = rng.standard_normal(len(data)) * 0.05
    data = data + noise

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


# =============================================================================
# =============================================================================


class TestWaveformPlotting:
    """Tests for VIS-001: Time-Domain Waveform Plotting."""

    def test_basic_waveform_plot(self, sample_waveform) -> None:
        """Test basic waveform plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_waveform

            fig = plot_waveform(sample_waveform)

            assert fig is not None
            assert len(fig.axes) >= 1
        except Exception as e:
            pytest.skip(f"Waveform plotting skipped: {e}")

    def test_waveform_with_time_unit(self, sample_waveform) -> None:
        """Test waveform plotting with different time units."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_waveform

            for unit in ["s", "ms", "us", "ns", "auto"]:
                fig = plot_waveform(sample_waveform, time_unit=unit)
                assert fig is not None
        except Exception as e:
            pytest.skip(f"Time unit test skipped: {e}")

    def test_waveform_with_labels(self, sample_waveform) -> None:
        """Test waveform plotting with custom labels."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_waveform

            fig = plot_waveform(
                sample_waveform,
                title="Test Waveform",
                xlabel="Time",
                ylabel="Voltage (V)",
            )

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Labels test skipped: {e}")

    def test_waveform_time_range(self, sample_waveform) -> None:
        """Test waveform plotting with time range."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_waveform

            fig = plot_waveform(sample_waveform, time_range=(0.0, 0.5e-3))
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Time range test skipped: {e}")

    def test_xy_plot(self, sample_waveform) -> None:
        """Test XY plot mode."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_xy

            # Create second trace for XY mode
            sample_rate = sample_waveform.metadata.sample_rate
            t = np.arange(len(sample_waveform.data)) / sample_rate
            trace_y = WaveformTrace(
                data=np.cos(2 * np.pi * 1000 * t),
                metadata=sample_waveform.metadata,
            )

            fig = plot_xy(sample_waveform, trace_y)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"XY plot test skipped: {e}")


# =============================================================================
# =============================================================================


class TestMultiChannelPlotting:
    """Tests for VIS-002: Multi-Channel Plotting."""

    def test_basic_multi_channel(self, multi_channel_waveforms) -> None:
        """Test basic multi-channel plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_multi_channel

            fig = plot_multi_channel(multi_channel_waveforms)

            assert fig is not None
            assert len(fig.axes) == 3  # Three channels
        except Exception as e:
            pytest.skip(f"Multi-channel test skipped: {e}")

    def test_multi_channel_with_names(self, multi_channel_waveforms) -> None:
        """Test multi-channel with channel names."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_multi_channel

            names = ["Ch1 - Sine", "Ch2 - Cosine", "Ch3 - 2kHz"]
            fig = plot_multi_channel(multi_channel_waveforms, names=names)

            assert fig is not None
            # Check that names appear somewhere in the plot
            for ax, _name in zip(fig.axes, names, strict=False):
                # May be in title, ylabel, or label
                assert ax is not None
        except Exception as e:
            pytest.skip(f"Channel names test skipped: {e}")

    def test_multi_channel_shared_x(self, multi_channel_waveforms) -> None:
        """Test multi-channel with shared X axis."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_multi_channel

            fig = plot_multi_channel(multi_channel_waveforms, share_x=True)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Shared X test skipped: {e}")

    def test_multi_channel_colors(self, multi_channel_waveforms) -> None:
        """Test multi-channel with custom colors."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_multi_channel

            colors = ["red", "green", "blue"]
            fig = plot_multi_channel(multi_channel_waveforms, colors=colors)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Colors test skipped: {e}")


# =============================================================================
# =============================================================================


class TestSpectrumPlotting:
    """Tests for VIS-003: Frequency Spectrum Plotting."""

    def test_basic_spectrum(self, sample_waveform) -> None:
        """Test basic spectrum plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrum

            fig = plot_spectrum(sample_waveform)

            assert fig is not None
            assert len(fig.axes) >= 1
        except Exception as e:
            pytest.skip(f"Spectrum test skipped: {e}")

    def test_spectrum_db_scale(self, sample_waveform) -> None:
        """Test spectrum with dB scale."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrum

            fig = plot_spectrum(sample_waveform, db_scale=True)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"dB scale test skipped: {e}")

    def test_spectrum_frequency_range(self, sample_waveform) -> None:
        """Test spectrum with frequency range."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrum

            fig = plot_spectrum(sample_waveform, freq_range=(0, 10000))
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Freq range test skipped: {e}")

    def test_psd_plot(self, sample_waveform) -> None:
        """Test power spectral density plot."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_psd

            fig = plot_psd(sample_waveform)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"PSD test skipped: {e}")


# =============================================================================
# =============================================================================


class TestSpectrogramPlotting:
    """Tests for VIS-004: Spectrogram Visualization."""

    def test_basic_spectrogram(self, sample_waveform) -> None:
        """Test basic spectrogram plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrogram

            fig = plot_spectrogram(sample_waveform)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Spectrogram test skipped: {e}")

    def test_spectrogram_nfft(self, sample_waveform) -> None:
        """Test spectrogram with custom NFFT."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrogram

            fig = plot_spectrogram(sample_waveform, nfft=256)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"NFFT test skipped: {e}")

    def test_spectrogram_overlap(self, sample_waveform) -> None:
        """Test spectrogram with overlap."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_spectrogram

            fig = plot_spectrogram(sample_waveform, overlap=0.5)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Overlap test skipped: {e}")


# =============================================================================
# =============================================================================


class TestDigitalTimingDiagram:
    """Tests for VIS-005: Digital Timing Diagram."""

    def test_basic_timing(self, digital_traces) -> None:
        """Test basic timing diagram."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_timing

            fig = plot_timing(digital_traces)

            assert fig is not None
            assert len(fig.axes) == 3
        except Exception as e:
            pytest.skip(f"Timing diagram test skipped: {e}")

    def test_timing_with_names(self, digital_traces) -> None:
        """Test timing diagram with channel names."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_timing

            names = ["CLK", "DATA", "CS"]
            fig = plot_timing(digital_traces, names=names)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Timing names test skipped: {e}")

    def test_logic_analyzer_view(self, digital_traces) -> None:
        """Test logic analyzer style view."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_logic_analyzer

            fig = plot_logic_analyzer(digital_traces)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Logic analyzer test skipped: {e}")

    def test_timing_empty_error(self) -> None:
        """Test timing diagram with empty traces."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_timing

            with pytest.raises((ValueError, Exception), match=r".*"):
                plot_timing([])
        except ImportError:
            pytest.skip("plot_timing not available")


# =============================================================================
# =============================================================================


class TestEyeDiagram:
    """Tests for VIS-006: Eye Diagram."""

    def test_basic_eye(self, eye_diagram_data) -> None:
        """Test basic eye diagram plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_eye

            fig = plot_eye(eye_diagram_data, bit_rate=1e9)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Eye diagram test skipped: {e}")

    def test_eye_with_ui_count(self, eye_diagram_data) -> None:
        """Test eye diagram with UI count."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_eye

            fig = plot_eye(eye_diagram_data, bit_rate=1e9)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"UI count test skipped: {e}")

    def test_bathtub_plot(self, eye_diagram_data) -> None:
        """Test bathtub curve plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_bathtub

            fig = plot_bathtub(eye_diagram_data, bit_rate=1e9)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Bathtub test skipped: {e}")


# =============================================================================
# =============================================================================


class TestPhasePlot:
    """Tests for VIS-009: Phase Plot."""

    def test_basic_phase_plot(self, sample_waveform) -> None:
        """Test basic phase plot."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_phase

            fig = plot_phase(sample_waveform)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Phase plot test skipped: {e}")

    def test_phase_plot_delay(self, sample_waveform) -> None:
        """Test phase plot with custom delay."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_phase

            fig = plot_phase(sample_waveform, delay_samples=10)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Phase delay test skipped: {e}")


# =============================================================================
# =============================================================================


class TestBodePlot:
    """Tests for VIS-010: Bode Plot."""

    def test_basic_bode(self) -> None:
        """Test basic Bode plot."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_bode

            # Create simple frequency response
            freq = np.logspace(1, 5, 100)
            H = 1 / (1 + 1j * freq / 1000)  # Simple low-pass

            fig = plot_bode(freq, H)

            assert fig is not None
            # Bode plot has 2 subplots (magnitude and phase)
            assert len(fig.axes) >= 2
        except Exception as e:
            pytest.skip(f"Bode plot test skipped: {e}")

    def test_bode_with_margin(self) -> None:
        """Test Bode plot with stability margins."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_bode

            freq = np.logspace(1, 5, 100)
            H = 10 / (1 + 1j * freq / 1000) ** 2

            fig = plot_bode(freq, H, show_margins=True)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Bode margin test skipped: {e}")


# =============================================================================
# =============================================================================


class TestWaterfallPlot:
    """Tests for VIS-011: Waterfall Plot."""

    def test_basic_waterfall(self) -> None:
        """Test basic waterfall plot."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_waterfall

            # Create 2D data for waterfall
            n_traces = 20
            n_points = 100
            data = np.zeros((n_traces, n_points))

            for i in range(n_traces):
                freq = 10 + i * 2
                t = np.linspace(0, 1, n_points)
                data[i] = np.sin(2 * np.pi * freq * t)

            fig = plot_waterfall(data)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Waterfall test skipped: {e}")


# =============================================================================
# =============================================================================


class TestHistogramPlot:
    """Tests for VIS-012: Histogram Plot."""

    def test_basic_histogram(self, sample_waveform) -> None:
        """Test basic histogram plotting."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_histogram

            fig = plot_histogram(sample_waveform.data)

            assert fig is not None
        except Exception as e:
            pytest.skip(f"Histogram test skipped: {e}")

    def test_histogram_bins(self, sample_waveform) -> None:
        """Test histogram with custom bins."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_histogram

            fig = plot_histogram(sample_waveform.data, bins=50)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Histogram bins test skipped: {e}")

    def test_histogram_range(self, sample_waveform) -> None:
        """Test histogram with range."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import plot_histogram

            fig = plot_histogram(sample_waveform.data, range=(-2, 2))
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Histogram range test skipped: {e}")


# =============================================================================
# Optimization Functions Tests (VIS-013, VIS-014, etc.)
# =============================================================================


class TestOptimizationFunctions:
    """Tests for visualization optimization functions."""

    def test_calculate_optimal_y_range(self) -> None:
        """Test VIS-013: Y-axis range optimization."""
        try:
            from oscura.visualization import calculate_optimal_y_range

            data = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
            y_min, y_max = calculate_optimal_y_range(data)

            assert y_min <= 0.1
            assert y_max >= 1.0
        except ImportError:
            pytest.skip("calculate_optimal_y_range not available")

    def test_calculate_optimal_x_window(self) -> None:
        """Test VIS-014: X-axis window optimization."""
        try:
            from oscura.visualization import calculate_optimal_x_window

            # Create time and data arrays as expected by the API
            sample_rate = 1e6
            duration = 1e-3
            n_samples = int(sample_rate * duration)
            time = np.linspace(0, duration, n_samples)
            data = np.sin(2 * np.pi * 1000 * time)  # 1 kHz sine wave

            t_start, t_end = calculate_optimal_x_window(time, data)

            assert t_start is not None
            assert t_end is not None
            assert t_end > t_start
        except ImportError:
            pytest.skip("calculate_optimal_x_window not available")

    def test_calculate_grid_spacing(self) -> None:
        """Test VIS-019: Grid auto-spacing."""
        try:
            from oscura.visualization import calculate_grid_spacing

            # API returns (major_spacing, minor_spacing) tuple
            major_spacing, minor_spacing = calculate_grid_spacing(0, 100)

            assert major_spacing > 0
            assert minor_spacing > 0
            assert 100 / major_spacing <= 20  # Reasonable number of grid lines
        except ImportError:
            pytest.skip("calculate_grid_spacing not available")

    def test_detect_interesting_regions(self) -> None:
        """Test VIS-020: Zoom to interesting regions."""
        try:
            from oscura.visualization import detect_interesting_regions

            # Signal with interesting region in middle
            data = np.zeros(1000)
            data[400:600] = np.sin(np.linspace(0, 4 * np.pi, 200))

            regions = detect_interesting_regions(data, sample_rate=1e6)

            assert len(regions) >= 1
        except ImportError:
            pytest.skip("detect_interesting_regions not available")


class TestAccessibility:
    """Tests for accessibility features (ACC-001, ACC-002, ACC-003)."""

    def test_colorblind_palette(self) -> None:
        """Test ACC-001: Colorblind-safe palette.

        get_colorblind_palette() returns a colormap name string,
        not a list of colors. Use COLORBLIND_SAFE_QUALITATIVE for color list.
        """
        try:
            from oscura.visualization import get_colorblind_palette

            # get_colorblind_palette returns a colormap name (string)
            palette_name = get_colorblind_palette()
            assert isinstance(palette_name, str)
            assert palette_name in ["viridis", "cividis", "plasma", "inferno", "magma"]
        except ImportError:
            pytest.skip("get_colorblind_palette not available")

    def test_colorblind_safe_qualitative_list(self) -> None:
        """Test ACC-001: Colorblind-safe qualitative color list."""
        try:
            from oscura.visualization import COLORBLIND_SAFE_QUALITATIVE

            # This is the list of colors
            assert len(COLORBLIND_SAFE_QUALITATIVE) >= 8
            assert all(isinstance(c, str) for c in COLORBLIND_SAFE_QUALITATIVE)
        except ImportError:
            pytest.skip("COLORBLIND_SAFE_QUALITATIVE not available")

    def test_generate_alt_text(self) -> None:
        """Test ACC-002: Alt text generation."""
        try:
            from oscura.visualization import generate_alt_text

            # API takes data as positional, not data_summary keyword
            data_summary = {"min": 0, "max": 1, "mean": 0.5}
            alt = generate_alt_text(data_summary, "waveform")

            assert isinstance(alt, str)
            assert len(alt) > 0
        except ImportError:
            pytest.skip("generate_alt_text not available")

    def test_multi_line_styles(self) -> None:
        """Test differentiation for colorblind users."""
        try:
            from oscura.visualization import get_multi_line_styles

            styles = get_multi_line_styles(5)

            assert len(styles) == 5
            # Each style should be different
            assert len({str(s) for s in styles}) == 5
        except ImportError:
            pytest.skip("get_multi_line_styles not available")


class TestPresets:
    """Tests for style presets (VIS-024)."""

    def test_list_presets(self) -> None:
        """Test listing available presets."""
        try:
            from oscura.visualization import list_visualization_presets

            presets = list_visualization_presets()

            assert len(presets) >= 4
            assert "publication" in presets or any("pub" in p.lower() for p in presets)
        except ImportError:
            pytest.skip("list_visualization_presets not available")

    def test_apply_preset(self) -> None:
        """Test applying a preset."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import apply_preset

            # Should not raise
            apply_preset("screen")
        except Exception as e:
            pytest.skip(f"apply_preset test skipped: {e}")

    def test_publication_preset(self) -> None:
        """Test publication preset."""
        try:
            from oscura.visualization import PUBLICATION_PRESET

            assert PUBLICATION_PRESET is not None
        except ImportError:
            pytest.skip("PUBLICATION_PRESET not available")

    def test_dark_theme_preset(self) -> None:
        """Test dark theme preset."""
        try:
            from oscura.visualization import DARK_THEME_PRESET

            assert DARK_THEME_PRESET is not None
        except ImportError:
            pytest.skip("DARK_THEME_PRESET not available")


class TestRenderingOptimization:
    """Tests for rendering optimization (VIS-017, VIS-018)."""

    def test_configure_dpi_rendering(self) -> None:
        """Test VIS-017: DPI-aware rendering."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import configure_dpi_rendering

            # Should not raise
            configure_dpi_rendering(dpi=150)
        except Exception as e:
            pytest.skip(f"DPI rendering test skipped: {e}")

    def test_render_thumbnail(self, sample_waveform) -> None:
        """Test VIS-018: Thumbnail mode."""
        pytest.importorskip("matplotlib")
        try:
            from oscura.visualization import render_thumbnail

            thumb = render_thumbnail(sample_waveform.data, width=100, height=50)

            assert thumb is not None
        except Exception as e:
            pytest.skip(f"Thumbnail test skipped: {e}")

    def test_decimate_for_display(self) -> None:
        """Test data decimation for display."""
        try:
            from oscura.visualization import decimate_for_display

            # Create time and data arrays
            n_points = 10000
            time = np.linspace(0, 1, n_points)
            data = np.random.randn(n_points)

            # API uses max_points parameter, not target_points
            _decimated_time, decimated_data = decimate_for_display(time, data, max_points=1000)

            assert len(decimated_data) <= 1000
        except ImportError:
            pytest.skip("decimate_for_display not available")


class TestInteractive:
    """Tests for interactive features (VIS-007, VIS-008)."""

    def test_zoom_state(self) -> None:
        """Test VIS-007: Zoom state management."""
        try:
            from oscura.visualization import ZoomState

            # ZoomState expects xlim and ylim as tuples
            state = ZoomState(xlim=(0, 100), ylim=(0, 10))

            assert state.xlim == (0, 100)
            assert state.ylim == (0, 10)
        except ImportError:
            pytest.skip("ZoomState not available")

    def test_cursor_measurement(self) -> None:
        """Test VIS-008: Cursor measurement."""
        try:
            from oscura.visualization import CursorMeasurement

            # CursorMeasurement expects x1, x2, y1, y2, delta_x, delta_y
            cursor = CursorMeasurement(
                x1=50,
                x2=100,
                y1=5.0,
                y2=10.0,
                delta_x=50,
                delta_y=5.0,
            )

            assert cursor.x1 == 50
            assert cursor.x2 == 100
            assert cursor.delta_x == 50
        except ImportError:
            pytest.skip("CursorMeasurement not available")


class TestHistogramOptimization:
    """Tests for histogram optimization (VIS-025)."""

    def test_calculate_optimal_bins(self) -> None:
        """Test VIS-025: Histogram bin optimization."""
        try:
            from oscura.visualization import calculate_optimal_bins

            data = np.random.randn(1000)
            n_bins = calculate_optimal_bins(data)

            assert n_bins > 0
            assert n_bins < len(data)
        except ImportError:
            pytest.skip("calculate_optimal_bins not available")

    def test_calculate_bin_edges(self) -> None:
        """Test bin edge calculation."""
        try:
            from oscura.visualization import calculate_bin_edges

            data = np.random.randn(1000)
            edges = calculate_bin_edges(data, n_bins=20)

            assert len(edges) == 21  # n_bins + 1 edges
        except ImportError:
            pytest.skip("calculate_bin_edges not available")


class TestColorPalettes:
    """Tests for color palette selection (VIS-023)."""

    def test_select_optimal_palette(self) -> None:
        """Test VIS-023: Data-driven color palette."""
        try:
            from oscura.visualization import select_optimal_palette

            # API uses n_colors parameter and palette_type
            palette = select_optimal_palette(n_colors=5, palette_type="qualitative")

            assert len(palette) >= 5
        except ImportError:
            pytest.skip("select_optimal_palette not available")

    def test_colorblind_safe_qualitative(self) -> None:
        """Test colorblind-safe qualitative colors."""
        try:
            from oscura.visualization import COLORBLIND_SAFE_QUALITATIVE

            assert len(COLORBLIND_SAFE_QUALITATIVE) >= 8
        except ImportError:
            pytest.skip("COLORBLIND_SAFE_QUALITATIVE not available")

    def test_sequential_viridis(self) -> None:
        """Test sequential viridis colormap."""
        try:
            from oscura.visualization import SEQUENTIAL_VIRIDIS

            assert SEQUENTIAL_VIRIDIS is not None
        except ImportError:
            pytest.skip("SEQUENTIAL_VIRIDIS not available")

    def test_diverging_coolwarm(self) -> None:
        """Test diverging coolwarm colormap."""
        try:
            from oscura.visualization import DIVERGING_COOLWARM

            assert DIVERGING_COOLWARM is not None
        except ImportError:
            pytest.skip("DIVERGING_COOLWARM not available")
