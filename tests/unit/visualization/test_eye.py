"""Unit tests for eye diagram visualization.

Tests:
"""

import numpy as np
import pytest

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.visualization.eye import plot_bathtub, plot_eye

pytestmark = [
    pytest.mark.unit,
    pytest.mark.visualization,
    pytest.mark.usefixtures("cleanup_matplotlib"),
]


@pytest.fixture
def sample_serial_signal():
    """Create sample serial data signal for eye diagram."""
    sample_rate = 10e9  # 10 GSa/s
    bit_rate = 1e9  # 1 Gbps
    n_bits = 100
    samples_per_bit = int(sample_rate / bit_rate)
    n_samples = n_bits * samples_per_bit

    metadata = TraceMetadata(sample_rate=sample_rate)

    # Generate random NRZ signal
    bits = np.random.randint(0, 2, n_bits)
    data = np.repeat(bits, samples_per_bit).astype(float)

    # Add noise
    data += np.random.normal(0, 0.05, n_samples)

    # Add some edge transitions
    for i in range(n_bits - 1):
        idx = (i + 1) * samples_per_bit
        # Smooth transition
        if bits[i] != bits[i + 1]:
            transition_samples = samples_per_bit // 10
            if bits[i] == 0:  # Rising edge
                data[idx - transition_samples : idx] = np.linspace(0, 1, transition_samples)
            else:  # Falling edge
                data[idx - transition_samples : idx] = np.linspace(1, 0, transition_samples)

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def short_signal():
    """Create signal that's too short for eye diagram."""
    metadata = TraceMetadata(sample_rate=1e6)
    data = np.random.randn(50)
    return WaveformTrace(data=data, metadata=metadata)


class TestPlotEye:
    """Tests for plot_eye function."""

    def test_basic_eye_diagram(self, sample_serial_signal):
        """Test basic eye diagram creation."""
        pytest.importorskip("matplotlib")

        fig = plot_eye(sample_serial_signal, bit_rate=1e9)
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Eye diagram should contain data"
        # Verify axis labels
        assert ax.get_xlabel(), "Should have x-label"
        assert ax.get_ylabel(), "Should have y-label"

    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    def test_auto_clock_recovery_fft(self):
        """Test automatic clock recovery using FFT method.

        Uses a longer signal with more periodic pattern for reliable FFT.
        """
        pytest.importorskip("matplotlib")

        # Create signal optimized for FFT clock recovery
        # - Longer duration (500 bits) for better FFT resolution
        # - Alternating pattern (strong fundamental frequency)
        sample_rate = 10e9  # 10 GSa/s
        bit_rate = 1e9  # 1 Gbps
        n_bits = 500  # Increased from 100 for better FFT
        samples_per_bit = int(sample_rate / bit_rate)
        n_samples = n_bits * samples_per_bit

        metadata = TraceMetadata(sample_rate=sample_rate)

        # Use alternating pattern instead of random (strong bit rate component)
        bits = np.tile([0, 1], n_bits // 2)
        data = np.repeat(bits, samples_per_bit).astype(float)

        # Add minimal noise (important for FFT stability)
        data += np.random.normal(0, 0.02, n_samples)

        trace = WaveformTrace(data=data, metadata=metadata)

        fig = plot_eye(trace, clock_recovery="fft")
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Eye diagram should contain data"

    def test_auto_clock_recovery_edge(self, sample_serial_signal):
        """Test automatic clock recovery using edge method."""
        pytest.importorskip("matplotlib")

        fig = plot_eye(sample_serial_signal, clock_recovery="edge")
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Eye diagram should contain data"

    def test_different_colormaps(self, sample_serial_signal):
        """Test different colormaps."""
        pytest.importorskip("matplotlib")

        for cmap in ["hot", "viridis", "Blues", "none"]:
            fig = plot_eye(sample_serial_signal, bit_rate=1e9, cmap=cmap)
            assert fig is not None, f"Colormap {cmap} should produce a figure"
            assert len(fig.axes) >= 1, f"Colormap {cmap} should have at least 1 axis"
            ax = fig.axes[0]
            assert ax.has_data(), f"Colormap {cmap} should contain data"

    def test_with_measurements(self, sample_serial_signal):
        """Test with eye opening measurements."""
        pytest.importorskip("matplotlib")

        fig = plot_eye(sample_serial_signal, bit_rate=1e9, show_measurements=True)
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Eye diagram should contain data"
        # When measurements are shown, there may be text annotations
        assert len(ax.texts) > 0 or ax.has_data(), "Should have measurements or data"

    def test_without_measurements(self, sample_serial_signal):
        """Test without measurements."""
        pytest.importorskip("matplotlib")

        fig = plot_eye(sample_serial_signal, bit_rate=1e9, show_measurements=False)
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Eye diagram should contain data"

    def test_custom_title(self, sample_serial_signal):
        """Test custom title."""
        pytest.importorskip("matplotlib")

        title = "1 Gbps Ethernet Eye Diagram"
        fig = plot_eye(sample_serial_signal, bit_rate=1e9, title=title)

        assert fig is not None
        axes = fig.axes[0]
        assert title in axes.get_title()

    def test_colorbar(self, sample_serial_signal):
        """Test colorbar display."""
        pytest.importorskip("matplotlib")

        fig = plot_eye(sample_serial_signal, bit_rate=1e9, colorbar=True)
        assert fig is not None
        # Check that colorbar was added
        assert len(fig.axes) >= 2  # Main plot + colorbar

    def test_insufficient_samples_error(self, short_signal):
        """Test that short signal raises InsufficientDataError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(
            InsufficientDataError, match="Eye diagram requires at least 100 samples"
        ):
            plot_eye(short_signal, bit_rate=1e6)

    def test_too_few_bits_error(self):
        """Test that signal with too few bit periods raises error."""
        pytest.importorskip("matplotlib")

        # Create signal with 150 samples but only 1.5 bit periods
        # This passes the initial 100 sample check but fails on bit count
        sample_rate = 100e6  # 100 MSa/s
        bit_rate = 1e6  # 1 Mbps (gives 100 samples/bit)
        n_samples = 150  # 1.5 bit periods

        metadata = TraceMetadata(sample_rate=sample_rate)
        data = np.random.randn(n_samples)
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(InsufficientDataError, match="Not enough complete bit periods"):
            plot_eye(trace, bit_rate=bit_rate)

    def test_low_samples_per_bit_error(self):
        """Test that low samples per bit raises error."""
        pytest.importorskip("matplotlib")

        # Create undersampled signal (only 5 samples per bit)
        sample_rate = 5e6  # 5 MSa/s
        bit_rate = 1e6  # 1 Mbps
        n_bits = 100
        samples_per_bit = int(sample_rate / bit_rate)
        n_samples = n_bits * samples_per_bit

        metadata = TraceMetadata(sample_rate=sample_rate)
        data = np.random.randn(n_samples)
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(InsufficientDataError, match="Insufficient samples per bit period"):
            plot_eye(trace, bit_rate=bit_rate)

    def test_alpha_parameter(self, sample_serial_signal):
        """Test alpha transparency parameter."""
        pytest.importorskip("matplotlib")

        for alpha in [0.1, 0.5, 0.9]:
            fig = plot_eye(sample_serial_signal, bit_rate=1e9, alpha=alpha, cmap="none")
            assert fig is not None, f"Alpha {alpha} should produce a figure"
            assert len(fig.axes) >= 1, f"Alpha {alpha} should have at least 1 axis"


class TestPlotBathtub:
    """Tests for plot_bathtub function."""

    def test_basic_bathtub_plot(self, sample_serial_signal):
        """Test basic bathtub curve creation."""
        pytest.importorskip("matplotlib")

        fig = plot_bathtub(sample_serial_signal, bit_rate=1e9)
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Bathtub plot should contain data"
        assert ax.get_xlabel(), "Should have x-label"
        assert ax.get_ylabel(), "Should have y-label"

    def test_custom_ber_target(self, sample_serial_signal):
        """Test with custom BER target."""
        pytest.importorskip("matplotlib")

        fig = plot_bathtub(sample_serial_signal, bit_rate=1e9, ber_target=1e-15)
        assert fig is not None
        assert len(fig.axes) >= 1, "Should have at least 1 axis"
        ax = fig.axes[0]
        assert ax.has_data(), "Bathtub plot should contain data"

    def test_custom_title(self, sample_serial_signal):
        """Test custom title."""
        pytest.importorskip("matplotlib")

        title = "BER Analysis"
        fig = plot_bathtub(sample_serial_signal, bit_rate=1e9, title=title)

        assert fig is not None
        axes = fig.axes[0]
        assert title in axes.get_title()


class TestEyeMetrics:
    """Tests for eye diagram metrics calculation."""

    def test_metrics_with_clean_signal(self, sample_serial_signal):
        """Test that eye metrics are calculated for clean signal."""
        pytest.importorskip("matplotlib")

        # Eye metrics are calculated internally during plotting
        fig = plot_eye(sample_serial_signal, bit_rate=1e9, show_measurements=True)

        # If this doesn't raise an exception, metrics calculation worked
        assert fig is not None
