"""Tests for SignalBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from oscura.builders import SignalBuilder
from oscura.core.types import WaveformTrace

pytestmark = pytest.mark.unit


class TestSignalBuilder:
    """Tests for the fluent SignalBuilder API."""

    def test_basic_sine_generation(self) -> None:
        """Test basic sine wave generation."""
        trace = (
            SignalBuilder(sample_rate=1e6, duration=0.01)
            .add_sine(frequency=1000, amplitude=1.0)
            .build()
        )

        assert isinstance(trace, WaveformTrace)
        assert trace.metadata.sample_rate == 1e6
        assert len(trace.data) == 10000  # 0.01s * 1MHz

    def test_sine_with_noise(self) -> None:
        """Test sine wave with added noise."""
        trace = (
            SignalBuilder(sample_rate=1e6, duration=0.01)
            .add_sine(frequency=1000, amplitude=1.0)
            .add_noise(snr_db=40)
            .build()
        )

        # Verify noise was added (signal should not be pure sine)
        data = trace.data
        assert np.std(data) > 0  # Has variance
        # Check approximate SNR
        t = np.arange(len(data)) / 1e6
        ideal_sine = np.sin(2 * np.pi * 1000 * t)
        noise = data - ideal_sine
        actual_snr = 10 * np.log10(np.mean(ideal_sine**2) / np.mean(noise**2))
        assert 35 < actual_snr < 45  # Within 5 dB of target

    def test_uart_signal_generation(self) -> None:
        """Test UART signal generation (multi-channel)."""
        builder = SignalBuilder(sample_rate=10e6)
        builder.add_uart(baud_rate=115200, data=b"Hello", amplitude=3.3)
        channels = builder.build_channels()

        assert "uart" in channels
        uart_trace = channels["uart"]
        # Should have high and low levels
        assert np.max(uart_trace.data) > 3.0  # High level
        assert np.min(uart_trace.data) < 0.5  # Low level

    def test_spi_signal_generation(self) -> None:
        """Test SPI signal generation (multi-channel)."""
        builder = SignalBuilder(sample_rate=10e6)
        builder.add_spi(clock_freq=1e6, data_mosi=b"\x9f\x00")
        channels = builder.build_channels()

        # Should have 4 channels
        assert "sck" in channels
        assert "mosi" in channels
        assert "miso" in channels
        assert "cs" in channels

    def test_i2c_signal_generation(self) -> None:
        """Test I2C signal generation (multi-channel)."""
        builder = SignalBuilder(sample_rate=10e6)
        builder.add_i2c(clock_freq=100e3, address=0x50, data=b"\x00\x01")
        channels = builder.build_channels()

        assert "scl" in channels
        assert "sda" in channels

    def test_can_signal_generation(self) -> None:
        """Test CAN signal generation."""
        builder = SignalBuilder(sample_rate=10e6)
        builder.add_can(bitrate=500000, arbitration_id=0x123, data=b"\x01\x02\x03")
        channels = builder.build_channels()

        assert "can" in channels

    def test_multi_channel(self) -> None:
        """Test multi-channel signal generation with build_channels()."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.01)
        builder.add_sine(frequency=1000, channel="ch1")
        builder.add_square(frequency=500, channel="ch2")
        builder.add_triangle(frequency=250, channel="ch3")
        channels = builder.build_channels()

        assert len(channels) == 3
        assert "ch1" in channels
        assert "ch2" in channels
        assert "ch3" in channels
        # Each should be a WaveformTrace
        assert all(isinstance(t, WaveformTrace) for t in channels.values())

    def test_harmonics_addition(self) -> None:
        """Test harmonic distortion addition."""
        trace = (
            SignalBuilder(sample_rate=1e6, duration=0.01)
            .add_sine(frequency=1000, amplitude=1.0)
            .add_harmonics(fundamental=1000, thd_percent=5.0)
            .build()
        )

        # FFT should show harmonics
        data = trace.data
        fft_result = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1 / 1e6)

        # Find 2nd harmonic peak
        idx_2khz = np.argmin(np.abs(freqs - 2000))
        idx_1khz = np.argmin(np.abs(freqs - 1000))

        # 2nd harmonic should be visible (at least 3% of fundamental for 5% THD)
        assert fft_result[idx_2khz] > 0.03 * fft_result[idx_1khz]

    def test_build_returns_trace(self) -> None:
        """Test that build() returns WaveformTrace."""
        trace = SignalBuilder(sample_rate=1e6, duration=0.01).add_sine(frequency=1000).build()

        assert isinstance(trace, WaveformTrace)
        assert trace.metadata.sample_rate == 1e6

    def test_chained_configuration(self) -> None:
        """Test chained configuration methods."""
        trace = (
            SignalBuilder()
            .sample_rate(2e6)
            .duration(0.02)
            .description("Test signal")
            .add_sine(frequency=500)
            .build()
        )

        assert trace.metadata.sample_rate == 2e6

    def test_noise_without_signal_raises(self) -> None:
        """Test that adding noise without signal raises error."""
        builder = SignalBuilder()
        with pytest.raises(ValueError, match="does not exist"):
            builder.add_noise(snr_db=40)

    def test_chirp_generation(self) -> None:
        """Test chirp signal generation."""
        trace = SignalBuilder(sample_rate=1e6, duration=0.01).add_chirp(f0=1000, f1=10000).build()

        data = trace.data
        # Chirp should sweep through frequencies
        # Check instantaneous frequency increases
        from scipy.signal import hilbert

        analytic = hilbert(data)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * 1e6 / (2 * np.pi)

        # First half should have lower frequency than second half
        assert np.mean(inst_freq[: len(inst_freq) // 2]) < np.mean(inst_freq[len(inst_freq) // 2 :])

    def test_multitone_generation(self) -> None:
        """Test multi-tone signal generation."""
        freqs = [1000, 2000, 3000]
        trace = (
            SignalBuilder(sample_rate=1e6, duration=0.01).add_multitone(frequencies=freqs).build()
        )

        data = trace.data
        fft_result = np.abs(np.fft.rfft(data))
        fft_freqs = np.fft.rfftfreq(len(data), 1 / 1e6)

        # All three frequencies should have peaks
        for f in freqs:
            idx = np.argmin(np.abs(fft_freqs - f))
            # Should be a local maximum
            assert fft_result[idx] > fft_result[idx - 2]
            assert fft_result[idx] > fft_result[idx + 2]
