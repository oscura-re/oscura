"""Tests for new high-level API additions.

Tests the following new APIs:
- SignalBuilder (fluent signal generation)
- quick_spectral (one-call spectral analysis)
- auto_decode (unified protocol decoding)
- smart_filter (intelligent filtering)
- reverse_engineer_signal (complete RE workflow)

Phase 0.2 Updates:
- SignalBuilder.build() now returns WaveformTrace instead of GeneratedSignal
- Tests updated to reflect new API
"""

from __future__ import annotations

import numpy as np
import pytest

import oscura as osc
from oscura.builders import SignalBuilder
from oscura.convenience import SpectralMetrics, quick_spectral
from oscura.core.types import TraceMetadata, WaveformTrace

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


class TestQuickSpectral:
    """Tests for the quick_spectral convenience function."""

    def test_basic_spectral_analysis(self) -> None:
        """Test basic spectral analysis."""
        # Generate clean sine wave
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 1000
        data = np.sin(2 * np.pi * fundamental * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace, fundamental=fundamental)

        assert isinstance(metrics, SpectralMetrics)
        # Clean sine should have excellent metrics
        assert metrics.thd_db < -60 or np.isnan(metrics.thd_db)  # Very low THD
        assert metrics.fundamental_freq == pytest.approx(fundamental, rel=0.1)

    def test_spectral_with_harmonics(self) -> None:
        """Test spectral analysis with harmonic distortion."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 1000

        # Add 1% THD
        data = np.sin(2 * np.pi * fundamental * t)
        data += 0.01 * np.sin(2 * np.pi * 2 * fundamental * t)  # 2nd harmonic

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace, fundamental=fundamental)

        # THD should be around -40 dB (1%)
        assert -50 < metrics.thd_db < -30

    def test_auto_fundamental_detection(self) -> None:
        """Test automatic fundamental frequency detection."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 2500

        data = np.sin(2 * np.pi * fundamental * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace)  # No fundamental specified

        # Should auto-detect fundamental
        assert 2000 < metrics.fundamental_freq < 3000

    def test_returns_all_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        data = np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace)

        # All fields should be present
        assert hasattr(metrics, "thd_db")
        assert hasattr(metrics, "thd_percent")
        assert hasattr(metrics, "snr_db")
        assert hasattr(metrics, "sinad_db")
        assert hasattr(metrics, "enob")
        assert hasattr(metrics, "sfdr_db")
        assert hasattr(metrics, "fundamental_freq")
        assert hasattr(metrics, "fundamental_mag_db")
        assert hasattr(metrics, "noise_floor_db")


class TestSmartFilter:
    """Tests for the smart_filter convenience function."""

    def test_noise_filtering(self) -> None:
        """Test general noise filtering."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate

        # Clean signal with noise
        clean = np.sin(2 * np.pi * 1000 * t)
        noisy = clean + 0.3 * np.random.randn(n_samples)

        trace = WaveformTrace(
            data=noisy, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        filtered = osc.smart_filter(trace, target="noise")

        # Filtered should be cleaner (lower variance after removing trend)
        from scipy import signal

        # Detrend both
        detrended_original = signal.detrend(noisy)
        detrended_filtered = signal.detrend(filtered.data)

        # Filtered should have lower high-frequency content
        orig_hf = np.std(np.diff(noisy))
        filt_hf = np.std(np.diff(filtered.data))
        assert filt_hf < orig_hf

    def test_60hz_hum_filtering(self) -> None:
        """Test 60 Hz hum filtering."""
        sample_rate = 10000
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate

        # Signal with 60 Hz hum
        signal_clean = np.sin(2 * np.pi * 1000 * t)
        hum = 0.2 * np.sin(2 * np.pi * 60 * t)
        noisy = signal_clean + hum

        trace = WaveformTrace(
            data=noisy, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        filtered = osc.smart_filter(trace, target="60hz_hum")

        # Check 60 Hz component is reduced
        fft_orig = np.abs(np.fft.rfft(noisy))
        fft_filt = np.abs(np.fft.rfft(filtered.data))
        freqs = np.fft.rfftfreq(n_samples, 1 / sample_rate)

        idx_60 = np.argmin(np.abs(freqs - 60))

        # 60 Hz should be significantly reduced
        assert fft_filt[idx_60] < fft_orig[idx_60] * 0.5


class TestReverseEngineerSignal:
    """Tests for the reverse_engineer_signal workflow."""

    def test_basic_workflow(self) -> None:
        """Test basic reverse engineering workflow."""
        # Generate UART-like signal
        sample_rate = 1e6
        baud_rate = 19200
        samples_per_bit = int(sample_rate / baud_rate)

        # Create simple pattern: idle, start, data, stop
        bits = [1] * 100  # Idle
        for _ in range(5):  # 5 bytes
            bits.append(0)  # Start bit
            for i in range(8):  # Data bits
                bits.append((0xAA >> i) & 1)
            bits.append(1)  # Stop bit
            bits.extend([1] * 10)  # Gap

        # Expand to samples
        signal_data = []
        for bit in bits:
            signal_data.extend([bit * 3.3] * samples_per_bit)

        signal_data = np.array(signal_data) + 0.05 * np.random.randn(len(signal_data))

        trace = WaveformTrace(
            data=signal_data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        result = osc.workflows.reverse_engineer_signal(trace)

        # Should detect something
        assert result.baud_rate > 0
        assert len(result.bit_stream) > 100
        assert result.confidence > 0


class TestTopLevelExports:
    """Test that all new APIs are accessible at top level."""

    def test_signal_builder_accessible(self) -> None:
        """Test SignalBuilder is accessible via tk namespace."""
        assert hasattr(osc, "SignalBuilder")
        builder = osc.SignalBuilder()
        assert isinstance(builder, SignalBuilder)

    def test_convenience_functions_accessible(self) -> None:
        """Test convenience functions are accessible."""
        assert hasattr(osc, "quick_spectral")
        assert hasattr(osc, "auto_decode")
        assert hasattr(osc, "smart_filter")

    def test_workflow_accessible(self) -> None:
        """Test reverse_engineer_signal is accessible."""
        assert hasattr(osc, "reverse_engineer_signal")
        assert hasattr(osc.workflows, "reverse_engineer_signal")

    def test_discovery_functions_accessible(self) -> None:
        """Test discovery functions are accessible."""
        assert hasattr(osc, "characterize_signal")
        assert hasattr(osc, "find_anomalies")
        assert hasattr(osc, "assess_data_quality")
