"""Tests for convenience.py - High-level one-call analysis functions.

Tests:
- quick_spectral: Complete spectral metrics in one call
- auto_decode: Auto-detect and decode protocols
- smart_filter: Intelligent filtering with auto-detection
- All edge cases and error conditions
"""

import numpy as np
import pytest

from oscura import convenience
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace


class TestQuickSpectral:
    """Test quick_spectral function."""

    def test_basic_spectral_analysis(self) -> None:
        """Test basic spectral analysis with known frequency."""
        # Create 1kHz sine wave
        sample_rate = 100_000.0
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 1000.0
        signal = np.sin(2 * np.pi * freq * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        result = convenience.quick_spectral(trace, fundamental=freq)

        # Check all metrics present
        assert isinstance(result, convenience.SpectralMetrics)
        assert result.fundamental_freq == pytest.approx(freq, rel=0.01)
        assert not np.isnan(result.thd_db)
        assert not np.isnan(result.snr_db)
        assert not np.isnan(result.sinad_db)
        assert not np.isnan(result.enob)
        assert not np.isnan(result.sfdr_db)
        assert not np.isnan(result.fundamental_mag_db)
        assert not np.isnan(result.noise_floor_db)

        # THD % should be derivable from THD dB
        expected_pct = 100 * 10 ** (result.thd_db / 20)
        assert result.thd_percent == pytest.approx(expected_pct, rel=0.01)

    def test_auto_detect_fundamental(self) -> None:
        """Test fundamental frequency auto-detection."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        freq = 2500.0  # Dominant frequency
        signal = 0.8 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        result = convenience.quick_spectral(trace, fundamental=None)  # Auto-detect

        # Should detect fundamental within 5%
        assert result.fundamental_freq == pytest.approx(freq, rel=0.05)

    def test_with_harmonics(self) -> None:
        """Test with different harmonic counts."""
        sample_rate = 100_000.0
        t = np.linspace(0, 0.1, 10000)
        signal = np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        result_5 = convenience.quick_spectral(trace, fundamental=1000, n_harmonics=5)
        result_20 = convenience.quick_spectral(trace, fundamental=1000, n_harmonics=20)

        # Both should complete
        assert result_5.thd_db is not None
        assert result_20.thd_db is not None

    def test_different_windows(self) -> None:
        """Test with different window functions."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        signal = np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        for window in ["hann", "hamming", "blackman"]:
            result = convenience.quick_spectral(trace, fundamental=1000, window=window)
            assert result.fundamental_freq == 1000.0


class TestAutoDecode:
    """Test auto_decode function."""

    def test_uart_decode_digital(self) -> None:
        """Test UART decoding with digital trace."""
        # Create simple digital trace (all high)
        data = np.ones(1000, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=115200, channel="test"))

        result = convenience.auto_decode(trace, protocol="UART")

        assert isinstance(result, convenience.DecodeResult)
        assert result.protocol == "UART"
        assert isinstance(result.frames, list)
        assert result.confidence == 1.0  # Explicit protocol
        assert result.baud_rate == 115200  # Default config
        assert "data_bits" in result.config

    def test_uart_decode_waveform(self) -> None:
        """Test UART decoding with waveform trace (auto-conversion)."""
        sample_rate = 1_000_000.0
        t = np.linspace(0, 0.001, 1000)
        signal = np.ones_like(t) * 3.3  # High level

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        result = convenience.auto_decode(trace, protocol="UART")

        assert result.protocol == "UART"
        assert isinstance(result.frames, list)

    def test_spi_decode(self) -> None:
        """Test SPI decoding."""
        data = np.ones(1000, dtype=bool)
        trace = DigitalTrace(
            data=data, metadata=TraceMetadata(sample_rate=1_000_000, channel="test")
        )

        result = convenience.auto_decode(trace, protocol="SPI")

        assert result.protocol == "SPI"
        assert "clock_polarity" in result.config
        assert "clock_phase" in result.config

    def test_i2c_decode(self) -> None:
        """Test I2C decoding."""
        data = np.ones(1000, dtype=bool)
        trace = DigitalTrace(
            data=data, metadata=TraceMetadata(sample_rate=1_000_000, channel="test")
        )

        result = convenience.auto_decode(trace, protocol="I2C")

        assert result.protocol == "I2C"
        assert "clock_rate" in result.config

    def test_can_decode(self) -> None:
        """Test CAN decoding."""
        data = np.ones(1000, dtype=bool)
        trace = DigitalTrace(
            data=data, metadata=TraceMetadata(sample_rate=2_000_000, channel="test")
        )

        result = convenience.auto_decode(trace, protocol="CAN")

        assert result.protocol == "CAN"
        assert "baud_rate" in result.config

    def test_unsupported_protocol(self) -> None:
        """Test unsupported protocol handling."""
        data = np.ones(100, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=100_000, channel="test"))

        result = convenience.auto_decode(trace, protocol="UNKNOWN_PROTO")

        assert result.protocol == "UNKNOWN_PROTO"
        assert len(result.errors) > 0
        assert "Unsupported protocol" in result.errors[0]

    def test_auto_detect_protocol_digital(self) -> None:
        """Test auto protocol detection with digital trace."""
        data = np.ones(100, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=100_000, channel="test"))

        result = convenience.auto_decode(trace, protocol=None)

        # Should default to UART for digital traces
        assert result.protocol == "UART"

    def test_statistics_calculation(self) -> None:
        """Test statistics are calculated correctly."""
        data = np.ones(100, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=100_000, channel="test"))

        result = convenience.auto_decode(trace, protocol="UART")

        assert "total_frames" in result.statistics
        assert "error_frames" in result.statistics
        assert "error_rate" in result.statistics

    def test_min_confidence_parameter(self) -> None:
        """Test min_confidence parameter."""
        sample_rate = 100_000.0
        t = np.linspace(0, 0.01, 1000)
        signal = np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        result = convenience.auto_decode(trace, protocol=None, min_confidence=0.8)

        # Should still work, might fall back to defaults
        assert result.protocol is not None


class TestSmartFilter:
    """Test smart_filter function."""

    def test_noise_filter(self) -> None:
        """Test general noise filtering."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="noise", strength=1.0)

        assert isinstance(filtered, WaveformTrace)
        assert len(filtered.data) <= len(trace.data)  # Median filter may trim edges

    def test_high_freq_filter(self) -> None:
        """Test high frequency removal."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 10000 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="high_freq", strength=1.0)

        assert isinstance(filtered, WaveformTrace)
        # High frequency component should be attenuated

    def test_low_freq_filter(self) -> None:
        """Test DC and low frequency removal."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        signal = 2.0 + 0.1 * t + np.sin(2 * np.pi * 1000 * t)  # DC + drift + signal

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="low_freq", strength=1.0)

        assert isinstance(filtered, WaveformTrace)

    def test_60hz_hum_filter(self) -> None:
        """Test 60 Hz power line interference removal."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="60hz_hum", strength=1.0)

        assert isinstance(filtered, WaveformTrace)

    def test_50hz_hum_filter(self) -> None:
        """Test 50 Hz power line interference removal."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="50hz_hum", strength=1.0)

        assert isinstance(filtered, WaveformTrace)

    def test_auto_detect_filter(self) -> None:
        """Test automatic noise type detection."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        # Add strong 60 Hz component
        signal = np.sin(2 * np.pi * 500 * t) + 1.0 * np.sin(2 * np.pi * 60 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        filtered = convenience.smart_filter(trace, target="auto")

        assert isinstance(filtered, WaveformTrace)

    def test_filter_strength_variations(self) -> None:
        """Test different filter strengths."""
        sample_rate = 50_000.0
        t = np.linspace(0, 0.1, 5000)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.2 * np.random.randn(len(t))

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        for strength in [0.0, 0.5, 1.0]:
            filtered = convenience.smart_filter(trace, target="noise", strength=strength)
            assert isinstance(filtered, WaveformTrace)

    def test_invalid_filter_target(self) -> None:
        """Test invalid filter target raises error."""
        sample_rate = 10_000.0
        signal = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000))

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        with pytest.raises(ValueError, match="Unknown filter target"):
            convenience.smart_filter(trace, target="invalid_target")  # type: ignore


class TestDetectNoiseType:
    """Test _detect_noise_type helper function."""

    def test_detect_60hz_hum(self) -> None:
        """Test detection of 60 Hz power line hum."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        # Strong 60 Hz component
        signal = 0.1 * np.sin(2 * np.pi * 500 * t) + 1.0 * np.sin(2 * np.pi * 60 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        noise_type = convenience._detect_noise_type(trace)

        assert noise_type == "60hz_hum"

    def test_detect_50hz_hum(self) -> None:
        """Test detection of 50 Hz power line hum."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        # Strong 50 Hz component
        signal = 0.1 * np.sin(2 * np.pi * 500 * t) + 1.0 * np.sin(2 * np.pi * 50 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        noise_type = convenience._detect_noise_type(trace)

        assert noise_type == "50hz_hum"

    def test_detect_low_freq(self) -> None:
        """Test detection of low frequency noise."""
        sample_rate = 10_000.0
        t = np.linspace(0, 1, 10000)
        # Strong low frequency component
        signal = 5.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        noise_type = convenience._detect_noise_type(trace)

        assert noise_type == "low_freq"

    def test_detect_general_noise(self) -> None:
        """Test detection of general noise."""
        sample_rate = 10_000.0
        t = np.linspace(0, 0.5, 5000)
        # Broadband noise
        signal = np.random.randn(len(t))

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        noise_type = convenience._detect_noise_type(trace)

        # Should default to general noise
        assert noise_type in ["noise", "high_freq", "low_freq"]


class TestGetDefaultProtocolConfig:
    """Test _get_default_protocol_config helper function."""

    def test_uart_defaults(self) -> None:
        """Test UART default configuration."""
        config = convenience._get_default_protocol_config("UART")

        assert config["baud_rate"] == 115200
        assert config["data_bits"] == 8
        assert config["parity"] == "none"
        assert config["stop_bits"] == 1

    def test_spi_defaults(self) -> None:
        """Test SPI default configuration."""
        config = convenience._get_default_protocol_config("SPI")

        assert config["clock_polarity"] == 0
        assert config["clock_phase"] == 0
        assert config["bit_order"] == "MSB"

    def test_i2c_defaults(self) -> None:
        """Test I2C default configuration."""
        config = convenience._get_default_protocol_config("I2C")

        assert config["clock_rate"] == 100000
        assert config["address_bits"] == 7

    def test_can_defaults(self) -> None:
        """Test CAN default configuration."""
        config = convenience._get_default_protocol_config("CAN")

        assert config["baud_rate"] == 500000
        assert config["sample_point"] == 0.75

    def test_unknown_protocol(self) -> None:
        """Test unknown protocol returns empty config."""
        config = convenience._get_default_protocol_config("UNKNOWN")

        assert config == {}


class TestDataclasses:
    """Test dataclass structures."""

    def test_spectral_metrics_creation(self) -> None:
        """Test SpectralMetrics dataclass creation."""
        metrics = convenience.SpectralMetrics(
            thd_db=-40.0,
            thd_percent=1.0,
            snr_db=60.0,
            sinad_db=58.0,
            enob=9.5,
            sfdr_db=65.0,
            fundamental_freq=1000.0,
            fundamental_mag_db=-3.0,
            noise_floor_db=-80.0,
        )

        assert metrics.thd_db == -40.0
        assert metrics.snr_db == 60.0
        assert metrics.fundamental_freq == 1000.0

    def test_decode_result_creation(self) -> None:
        """Test DecodeResult dataclass creation."""
        result = convenience.DecodeResult(
            protocol="UART",
            frames=[],
            confidence=0.95,
            baud_rate=115200.0,
            config={"data_bits": 8},
            errors=[],
            statistics={"total_frames": 10},
        )

        assert result.protocol == "UART"
        assert result.confidence == 0.95
        assert result.baud_rate == 115200.0
        assert result.statistics["total_frames"] == 10
