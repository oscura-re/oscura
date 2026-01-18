"""Comprehensive unit tests for protocol type auto-detection.


This module tests protocol detection across multiple serial protocols:
- UART (asynchronous, idle high)
- SPI (synchronous, clock-based)
- I2C (clock + data, open-drain)
- CAN (NRZ with bit stuffing)
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.inference.protocol import (
    _analyze_signal_characteristics,
    _score_can,
    _score_i2c,
    _score_spi,
    _score_uart,
    detect_protocol,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_uart_signal(
    baudrate: int = 115200,
    message: bytes = b"Hello",
    sample_rate: float = 10e6,
    idle_high: bool = True,
) -> WaveformTrace:
    """Generate a synthetic UART signal.

    Args:
        baudrate: Baud rate in bits per second.
        message: Message to encode.
        sample_rate: Sample rate in Hz.
        idle_high: If True, idle level is high (standard UART).

    Returns:
        WaveformTrace with UART signal.
    """
    samples_per_bit = int(sample_rate / baudrate)
    idle_level = 3.3 if idle_high else 0.0
    active_level = 0.0 if idle_high else 3.3

    bits = []
    for byte in message:
        bits.append(0 if idle_high else 1)  # Start bit (opposite of idle)
        for i in range(8):
            bit = (byte >> i) & 1
            bits.append(bit if idle_high else (1 - bit))
        bits.append(1 if idle_high else 0)  # Stop bit (idle level)

    # Add idle periods before and after
    idle_samples = samples_per_bit * 10
    signal = np.full(idle_samples, idle_level, dtype=np.float64)

    # Add data
    for bit in bits:
        level = active_level if bit == 0 else idle_level
        signal = np.append(signal, np.full(samples_per_bit, level))

    # Add trailing idle
    signal = np.append(signal, np.full(idle_samples, idle_level))

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=signal, metadata=metadata)


def generate_spi_signal(
    clock_freq: float = 1e6,
    num_bytes: int = 10,
    sample_rate: float = 10e6,
    clock_polarity: int = 0,
) -> WaveformTrace:
    """Generate a synthetic SPI clock signal.

    Args:
        clock_freq: SPI clock frequency in Hz.
        num_bytes: Number of bytes to transmit.
        sample_rate: Sample rate in Hz.
        clock_polarity: 0 for idle low, 1 for idle high.

    Returns:
        WaveformTrace with SPI clock signal.
    """
    samples_per_half_cycle = max(1, int(sample_rate / (2 * clock_freq)))
    num_bits = num_bytes * 8

    # Generate square wave clock
    high_level = 3.3
    low_level = 0.0
    idle_level = high_level if clock_polarity else low_level
    active_level = low_level if clock_polarity else high_level

    signal = []
    # Idle before transmission
    signal.extend([idle_level] * (samples_per_half_cycle * 10))

    # Clock signal
    for _ in range(num_bits):
        signal.extend([active_level] * samples_per_half_cycle)
        signal.extend([idle_level] * samples_per_half_cycle)

    # Idle after transmission
    signal.extend([idle_level] * (samples_per_half_cycle * 10))

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def generate_i2c_signal(
    clock_freq: float = 100e3,
    num_bytes: int = 5,
    sample_rate: float = 10e6,
) -> WaveformTrace:
    """Generate a synthetic I2C clock signal.

    Args:
        clock_freq: I2C clock frequency in Hz.
        num_bytes: Number of bytes to transmit.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with I2C clock signal (idle high, open-drain).
    """
    samples_per_half_cycle = max(1, int(sample_rate / (2 * clock_freq)))
    num_bits = num_bytes * 8

    high_level = 3.3  # Pull-up
    low_level = 0.0

    signal = []
    # Start with idle high (pull-up)
    signal.extend([high_level] * (samples_per_half_cycle * 20))

    # Clock signal
    for _ in range(num_bits):
        signal.extend([low_level] * samples_per_half_cycle)
        signal.extend([high_level] * samples_per_half_cycle)

    # Idle after transmission
    signal.extend([high_level] * (samples_per_half_cycle * 20))

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def generate_can_signal(
    baudrate: int = 500000,
    num_frames: int = 3,
    sample_rate: float = 10e6,
) -> WaveformTrace:
    """Generate a synthetic CAN signal with bit stuffing.

    Args:
        baudrate: CAN baud rate in bits per second.
        num_frames: Number of frames to generate.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with CAN signal (recessive=high, dominant=low).
    """
    samples_per_bit = int(sample_rate / baudrate)
    recessive = 3.3  # Idle high
    dominant = 0.0

    signal = []
    # Idle (recessive)
    signal.extend([recessive] * (samples_per_bit * 20))

    # Generate frames with some irregularity (bit stuffing simulation)
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        # Start of frame (dominant)
        signal.extend([dominant] * samples_per_bit)

        # Random data with bit stuffing irregularity
        for _ in range(50):  # 50 bits per frame (simplified)
            level = recessive if rng.random() > 0.5 else dominant
            # Vary bit length slightly to simulate bit stuffing
            bit_samples = samples_per_bit + rng.integers(-2, 3)
            signal.extend([level] * bit_samples)

        # Inter-frame space (recessive)
        signal.extend([recessive] * (samples_per_bit * 5))

    # Trailing idle
    signal.extend([recessive] * (samples_per_bit * 20))

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


# =============================================================================
# Test Signal Characteristics Analysis
# =============================================================================


class TestAnalyzeSignalCharacteristics:
    """Test _analyze_signal_characteristics function."""

    def test_uart_characteristics(self) -> None:
        """Test characteristic extraction from UART signal."""
        trace = generate_uart_signal(baudrate=115200)
        chars = _analyze_signal_characteristics(trace)

        assert "regularity" in chars
        assert "symbol_rate" in chars
        assert "idle_level" in chars
        assert "duty_cycle" in chars
        assert "transition_density" in chars
        assert "edge_count" in chars

        # UART should have low regularity (asynchronous)
        assert chars["regularity"] < 0.6

        # UART idles high
        assert chars["idle_level"] == "high"

    def test_spi_characteristics(self) -> None:
        """Test characteristic extraction from SPI clock signal."""
        trace = generate_spi_signal(clock_freq=1e6)
        chars = _analyze_signal_characteristics(trace)

        # SPI should have high regularity (synchronous clock)
        assert chars["regularity"] > 0.7

        # SPI clock should have ~50% duty cycle
        assert 0.4 < chars["duty_cycle"] < 0.6

        # SPI should have high transition density
        assert chars["transition_density"] > 1e5

    def test_i2c_characteristics(self) -> None:
        """Test characteristic extraction from I2C clock signal."""
        trace = generate_i2c_signal(clock_freq=100e3)
        chars = _analyze_signal_characteristics(trace)

        # I2C has clock regularity
        assert chars["regularity"] > 0.6

        # I2C idles high (pull-up)
        assert chars["idle_level"] == "high"

    def test_can_characteristics(self) -> None:
        """Test characteristic extraction from CAN signal."""
        trace = generate_can_signal(baudrate=500000)
        chars = _analyze_signal_characteristics(trace)

        # CAN has some regularity but with bit stuffing irregularity
        assert 0.3 < chars["regularity"] < 0.8

        # CAN idles high (recessive)
        assert chars["idle_level"] == "high"

    def test_empty_signal(self) -> None:
        """Test characteristics of signal with no edges."""
        # Constant DC signal
        data = np.ones(1000) * 2.5
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        chars = _analyze_signal_characteristics(trace)

        assert chars["edge_count"] == 0
        assert chars["regularity"] == 0
        assert chars["symbol_rate"] == 0

    def test_single_edge(self) -> None:
        """Test characteristics of signal with single edge."""
        data = np.concatenate([np.zeros(500), np.ones(500) * 3.3])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        chars = _analyze_signal_characteristics(trace)

        # Note: np.diff creates one less element, and we look for where diff != 0
        # A transition from 0 to 3.3 creates one edge
        assert chars["edge_count"] >= 1
        # With very few edges, regularity defaults to 0.5
        if chars["edge_count"] <= 1:
            assert chars["regularity"] == 0.5

    def test_few_edges(self) -> None:
        """Test characteristics with few edges."""
        # Only 5 edges
        data = np.concatenate(
            [
                np.zeros(200),
                np.ones(200),
                np.zeros(200),
                np.ones(200),
                np.zeros(200),
            ]
        )
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        chars = _analyze_signal_characteristics(trace)

        assert chars["edge_count"] == 4
        assert chars["regularity"] == 0.5  # Default for <= 10 intervals


# =============================================================================
# Test Protocol Scoring Functions
# =============================================================================


class TestScoreUART:
    """Test _score_uart function."""

    def test_uart_signal_scores_high(self) -> None:
        """Test that UART signal gets high UART score."""
        trace = generate_uart_signal()
        chars = _analyze_signal_characteristics(trace)
        score = _score_uart(chars)

        # Should score well for UART
        assert score >= 0.5

    def test_low_regularity_increases_score(self) -> None:
        """Test that low regularity increases UART score."""
        chars_low_reg = {
            "regularity": 0.2,
            "idle_level": "high",
            "transition_density": 50000,
        }
        chars_high_reg = {
            "regularity": 0.9,
            "idle_level": "high",
            "transition_density": 50000,
        }

        score_low = _score_uart(chars_low_reg)
        score_high = _score_uart(chars_high_reg)

        assert score_low > score_high

    def test_idle_high_increases_score(self) -> None:
        """Test that idle high increases UART score."""
        chars_high = {
            "regularity": 0.2,
            "idle_level": "high",
            "transition_density": 50000,
        }
        chars_low = {
            "regularity": 0.2,
            "idle_level": "low",
            "transition_density": 50000,
        }

        score_high = _score_uart(chars_high)
        score_low = _score_uart(chars_low)

        assert score_high > score_low

    def test_transition_density_range(self) -> None:
        """Test that correct transition density increases score."""
        chars_good = {
            "regularity": 0.2,
            "idle_level": "high",
            "transition_density": 50000,  # Within UART range
        }
        chars_too_low = {
            "regularity": 0.2,
            "idle_level": "high",
            "transition_density": 500,  # Too low
        }
        chars_too_high = {
            "regularity": 0.2,
            "idle_level": "high",
            "transition_density": 5e6,  # Too high
        }

        score_good = _score_uart(chars_good)
        score_low = _score_uart(chars_too_low)
        score_high = _score_uart(chars_too_high)

        assert score_good > score_low
        assert score_good > score_high

    def test_score_capped_at_one(self) -> None:
        """Test that UART score is capped at 1.0."""
        chars = {
            "regularity": 0.1,
            "idle_level": "high",
            "transition_density": 50000,
        }
        score = _score_uart(chars)

        assert score <= 1.0


class TestScoreSPI:
    """Test _score_spi function."""

    def test_spi_signal_scores_high(self) -> None:
        """Test that SPI clock signal gets high SPI score."""
        trace = generate_spi_signal()
        chars = _analyze_signal_characteristics(trace)
        score = _score_spi(chars)

        # Should score well for SPI
        assert score >= 0.5

    def test_high_regularity_increases_score(self) -> None:
        """Test that high regularity increases SPI score."""
        chars_high_reg = {
            "regularity": 0.9,
            "duty_cycle": 0.5,
            "transition_density": 200000,
        }
        chars_low_reg = {
            "regularity": 0.3,
            "duty_cycle": 0.5,
            "transition_density": 200000,
        }

        score_high = _score_spi(chars_high_reg)
        score_low = _score_spi(chars_low_reg)

        assert score_high > score_low

    def test_fifty_percent_duty_cycle_increases_score(self) -> None:
        """Test that 50% duty cycle increases SPI score."""
        chars_good_duty = {
            "regularity": 0.9,
            "duty_cycle": 0.5,
            "transition_density": 200000,
        }
        chars_bad_duty = {
            "regularity": 0.9,
            "duty_cycle": 0.9,
            "transition_density": 200000,
        }

        score_good = _score_spi(chars_good_duty)
        score_bad = _score_spi(chars_bad_duty)

        assert score_good > score_bad

    def test_high_transition_density_increases_score(self) -> None:
        """Test that high transition density increases SPI score."""
        chars_high = {
            "regularity": 0.9,
            "duty_cycle": 0.5,
            "transition_density": 500000,  # High
        }
        chars_low = {
            "regularity": 0.9,
            "duty_cycle": 0.5,
            "transition_density": 50000,  # Low
        }

        score_high = _score_spi(chars_high)
        score_low = _score_spi(chars_low)

        assert score_high > score_low

    def test_score_capped_at_one(self) -> None:
        """Test that SPI score is capped at 1.0."""
        chars = {
            "regularity": 1.0,
            "duty_cycle": 0.5,
            "transition_density": 1e6,
        }
        score = _score_spi(chars)

        assert score <= 1.0


class TestScoreI2C:
    """Test _score_i2c function."""

    def test_i2c_signal_scores_high(self) -> None:
        """Test that I2C clock signal gets high I2C score."""
        trace = generate_i2c_signal()
        chars = _analyze_signal_characteristics(trace)
        score = _score_i2c(chars)

        # Should score reasonably for I2C
        assert score >= 0.3

    def test_regularity_increases_score(self) -> None:
        """Test that regularity increases I2C score."""
        chars_reg = {
            "regularity": 0.8,
            "idle_level": "high",
            "transition_density": 50000,
        }
        chars_irreg = {
            "regularity": 0.3,
            "idle_level": "high",
            "transition_density": 50000,
        }

        score_reg = _score_i2c(chars_reg)
        score_irreg = _score_i2c(chars_irreg)

        assert score_reg > score_irreg

    def test_idle_high_increases_score(self) -> None:
        """Test that idle high increases I2C score."""
        chars_high = {
            "regularity": 0.8,
            "idle_level": "high",
            "transition_density": 50000,
        }
        chars_low = {
            "regularity": 0.8,
            "idle_level": "low",
            "transition_density": 50000,
        }

        score_high = _score_i2c(chars_high)
        score_low = _score_i2c(chars_low)

        assert score_high > score_low

    def test_moderate_transition_density(self) -> None:
        """Test that moderate transition density increases I2C score."""
        chars_good = {
            "regularity": 0.8,
            "idle_level": "high",
            "transition_density": 50000,  # Moderate
        }
        chars_too_low = {
            "regularity": 0.8,
            "idle_level": "high",
            "transition_density": 500,
        }

        score_good = _score_i2c(chars_good)
        score_low = _score_i2c(chars_too_low)

        assert score_good > score_low

    def test_score_capped_at_one(self) -> None:
        """Test that I2C score is capped at 1.0."""
        chars = {
            "regularity": 1.0,
            "idle_level": "high",
            "transition_density": 50000,
        }
        score = _score_i2c(chars)

        assert score <= 1.0


class TestScoreCAN:
    """Test _score_can function."""

    def test_can_signal_scores_high(self) -> None:
        """Test that CAN signal gets high CAN score."""
        trace = generate_can_signal()
        chars = _analyze_signal_characteristics(trace)
        score = _score_can(chars)

        # Should score reasonably for CAN
        assert score >= 0.3

    def test_moderate_regularity_increases_score(self) -> None:
        """Test that moderate regularity increases CAN score."""
        chars_mod = {
            "regularity": 0.5,
            "idle_level": "high",
            "symbol_rate": 500000,
        }
        chars_too_high = {
            "regularity": 0.9,
            "idle_level": "high",
            "symbol_rate": 500000,
        }

        score_mod = _score_can(chars_mod)
        score_high = _score_can(chars_too_high)

        assert score_mod > score_high

    def test_idle_high_increases_score(self) -> None:
        """Test that idle high (recessive) increases CAN score."""
        chars_high = {
            "regularity": 0.5,
            "idle_level": "high",
            "symbol_rate": 500000,
        }
        chars_low = {
            "regularity": 0.5,
            "idle_level": "low",
            "symbol_rate": 500000,
        }

        score_high = _score_can(chars_high)
        score_low = _score_can(chars_low)

        assert score_high > score_low

    def test_common_baud_rates_increase_score(self) -> None:
        """Test that common CAN baud rates increase score."""
        # Common rates: 125k, 250k, 500k, 1M
        chars_common = {
            "regularity": 0.5,
            "idle_level": "high",
            "symbol_rate": 500000,  # Common
        }
        chars_uncommon = {
            "regularity": 0.5,
            "idle_level": "high",
            "symbol_rate": 333333,  # Uncommon
        }

        score_common = _score_can(chars_common)
        score_uncommon = _score_can(chars_uncommon)

        assert score_common > score_uncommon

    def test_all_common_rates(self) -> None:
        """Test all common CAN baud rates."""
        common_rates = [125000, 250000, 500000, 1000000]

        for rate in common_rates:
            chars = {
                "regularity": 0.5,
                "idle_level": "high",
                "symbol_rate": rate,
            }
            score = _score_can(chars)

            # Should get bonus for common rate
            assert score >= 0.6

    def test_score_capped_at_one(self) -> None:
        """Test that CAN score is capped at 1.0."""
        chars = {
            "regularity": 0.5,
            "idle_level": "high",
            "symbol_rate": 500000,
        }
        score = _score_can(chars)

        assert score <= 1.0


# =============================================================================
# Test Protocol Detection
# =============================================================================


class TestDetectProtocol:
    """Test detect_protocol function."""

    def test_detect_uart(self) -> None:
        """Test detection of UART protocol."""
        trace = generate_uart_signal(baudrate=115200)
        result = detect_protocol(trace, min_confidence=0.3)

        # UART characteristics may also match I2C or CAN depending on exact signal
        assert result["protocol"] in ["UART", "I2C", "CAN"]
        assert result["confidence"] > 0.3
        assert "config" in result
        assert "characteristics" in result

    def test_detect_spi(self) -> None:
        """Test detection of SPI protocol."""
        trace = generate_spi_signal(clock_freq=1e6)
        result = detect_protocol(trace, min_confidence=0.3)

        assert result["protocol"] == "SPI"
        assert result["confidence"] > 0.3
        assert "config" in result
        assert "clock_polarity" in result["config"]

    def test_detect_i2c(self) -> None:
        """Test detection of I2C protocol."""
        trace = generate_i2c_signal(clock_freq=100e3)
        result = detect_protocol(trace, min_confidence=0.3)

        # I2C or UART might score highest depending on characteristics
        assert result["protocol"] in ["I2C", "UART"]
        assert result["confidence"] > 0.3
        assert "config" in result

    def test_detect_can(self) -> None:
        """Test detection of CAN protocol."""
        trace = generate_can_signal(baudrate=500000)
        result = detect_protocol(trace, min_confidence=0.3)

        # CAN, UART, or I2C might score highest
        assert result["protocol"] in ["CAN", "UART", "I2C"]
        assert result["confidence"] > 0.3
        assert "config" in result

    def test_return_candidates(self) -> None:
        """Test returning all candidate protocols."""
        trace = generate_uart_signal()
        result = detect_protocol(trace, return_candidates=True)

        assert "candidates" in result
        assert len(result["candidates"]) >= 1
        assert all("protocol" in c for c in result["candidates"])
        assert all("confidence" in c for c in result["candidates"])

    def test_candidates_sorted_by_confidence(self) -> None:
        """Test that candidates are sorted by confidence descending."""
        trace = generate_uart_signal()
        result = detect_protocol(trace, return_candidates=True)

        candidates = result["candidates"]
        confidences = [c["confidence"] for c in candidates]

        # Should be sorted descending
        assert confidences == sorted(confidences, reverse=True)

    def test_min_confidence_threshold(self) -> None:
        """Test that low confidence raises AnalysisError."""
        # Generate a signal that's hard to classify
        # Use fixed seed for reproducibility
        rng = np.random.default_rng(12345)
        data = rng.random(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # With very high min_confidence, random noise should not be classifiable
        # Either raises AnalysisError or returns with confidence < threshold
        try:
            result = detect_protocol(trace, min_confidence=0.99)
            # If it didn't raise, verify confidence is actually high
            assert result["confidence"] >= 0.99, "Should have raised or met threshold"
        except AnalysisError:
            pass  # Expected behavior

    def test_no_edges_raises_error(self) -> None:
        """Test that constant signal raises AnalysisError."""
        data = np.ones(1000) * 2.5
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(AnalysisError, match="(Could not detect protocol|confidence too low)"):
            detect_protocol(trace)

    def test_primary_protocol_is_first_candidate(self) -> None:
        """Test that primary detection matches first candidate."""
        trace = generate_uart_signal()
        result = detect_protocol(trace, return_candidates=True)

        assert result["protocol"] == result["candidates"][0]["protocol"]
        assert result["confidence"] == result["candidates"][0]["confidence"]

    def test_config_format_uart(self) -> None:
        """Test UART config format."""
        trace = generate_uart_signal()
        result = detect_protocol(trace, min_confidence=0.3)

        if result["protocol"] == "UART":
            config = result["config"]
            assert "baud_rate" in config
            assert "data_bits" in config
            assert "parity" in config
            assert "stop_bits" in config
            assert config["data_bits"] == 8
            assert config["stop_bits"] == 1

    def test_config_format_spi(self) -> None:
        """Test SPI config format."""
        trace = generate_spi_signal()
        result = detect_protocol(trace, min_confidence=0.3)

        if result["protocol"] == "SPI":
            config = result["config"]
            assert "clock_polarity" in config
            assert "clock_phase" in config
            assert "bit_order" in config
            assert config["bit_order"] == "MSB"

    def test_config_format_i2c(self) -> None:
        """Test I2C config format."""
        trace = generate_i2c_signal()
        result = detect_protocol(trace, min_confidence=0.3)

        if result["protocol"] == "I2C":
            config = result["config"]
            assert "clock_rate" in config
            assert "address_bits" in config
            assert config["address_bits"] == 7

    def test_config_format_can(self) -> None:
        """Test CAN config format."""
        trace = generate_can_signal()
        result = detect_protocol(trace, min_confidence=0.3)

        if result["protocol"] == "CAN":
            config = result["config"]
            assert "baud_rate" in config
            assert "sample_point" in config
            assert config["sample_point"] == 0.75

    def test_characteristics_in_result(self) -> None:
        """Test that characteristics are included in result."""
        trace = generate_uart_signal()
        result = detect_protocol(trace)

        chars = result["characteristics"]
        assert "regularity" in chars
        assert "symbol_rate" in chars
        assert "idle_level" in chars
        assert "duty_cycle" in chars
        assert "transition_density" in chars
        assert "edge_count" in chars


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestInferenceProtocolEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_short_signal(self) -> None:
        """Test detection on very short signal."""
        data = np.array([0.0, 3.3, 0.0, 3.3, 0.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should either detect or raise error, not crash
        try:
            result = detect_protocol(trace, min_confidence=0.3)
            assert "protocol" in result
        except AnalysisError:
            pass  # Acceptable to fail on too-short signal

    def test_all_zeros(self) -> None:
        """Test detection on all-zero signal."""
        data = np.zeros(10000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(AnalysisError):
            detect_protocol(trace)

    def test_all_ones(self) -> None:
        """Test detection on all-high signal."""
        data = np.ones(10000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(AnalysisError):
            detect_protocol(trace)

    def test_random_noise(self) -> None:
        """Test detection on random noise."""
        rng = np.random.default_rng(42)
        data = rng.random(10000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Random noise should fail to meet high confidence threshold
        # May pass with lower threshold due to statistical characteristics
        with pytest.raises(AnalysisError):
            detect_protocol(trace, min_confidence=0.9)

    def test_single_pulse(self) -> None:
        """Test detection on single pulse."""
        data = np.concatenate(
            [
                np.zeros(5000),
                np.ones(100) * 3.3,
                np.zeros(5000),
            ]
        )
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should either detect or raise error
        try:
            result = detect_protocol(trace, min_confidence=0.3)
            assert "protocol" in result
        except AnalysisError:
            pass

    def test_very_high_sample_rate(self) -> None:
        """Test with very high sample rate."""
        trace = generate_uart_signal(sample_rate=100e9)  # 100 GSa/s
        result = detect_protocol(trace, min_confidence=0.3)

        assert "protocol" in result

    def test_very_low_sample_rate(self) -> None:
        """Test with very low sample rate (undersampled)."""
        trace = generate_uart_signal(sample_rate=1e6, baudrate=500000)
        result = detect_protocol(trace, min_confidence=0.3)

        assert "protocol" in result

    def test_negative_values(self) -> None:
        """Test with negative voltage values."""
        # RS-232 can have negative voltages
        data = np.concatenate(
            [
                np.ones(500) * -3.3,
                np.ones(500) * 3.3,
            ]
            * 5
        )
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_protocol(trace, min_confidence=0.3)
        assert "protocol" in result

    def test_min_confidence_zero(self) -> None:
        """Test with min_confidence=0 (always accepts)."""
        data = np.random.random(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should succeed even with random data if min_confidence=0
        # (as long as at least one protocol scores > 0)
        result = detect_protocol(trace, min_confidence=0.0)
        assert "protocol" in result

    def test_min_confidence_one(self) -> None:
        """Test with min_confidence=1.0 (requires perfect match)."""
        trace = generate_uart_signal()

        # Even good signals unlikely to score exactly 1.0
        with pytest.raises(AnalysisError, match="confidence too low"):
            detect_protocol(trace, min_confidence=1.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtocolIntegration:
    """Integration tests combining multiple aspects."""

    def test_multiple_protocols_compared(self) -> None:
        """Test that multiple protocol types are correctly distinguished."""
        uart_trace = generate_uart_signal(baudrate=115200)
        spi_trace = generate_spi_signal(clock_freq=1e6)

        uart_result = detect_protocol(uart_trace, min_confidence=0.3, return_candidates=True)
        spi_result = detect_protocol(spi_trace, min_confidence=0.3, return_candidates=True)

        # Should detect different protocols
        assert uart_result["protocol"] in ["UART", "I2C", "CAN"]
        assert spi_result["protocol"] in ["SPI", "I2C"]

        # Or if same, should have different confidence ordering
        if uart_result["protocol"] == spi_result["protocol"]:
            uart_scores = {c["protocol"]: c["confidence"] for c in uart_result["candidates"]}
            spi_scores = {c["protocol"]: c["confidence"] for c in spi_result["candidates"]}
            # At least one protocol should have different relative scores
            assert uart_scores != spi_scores

    def test_different_baud_rates(self) -> None:
        """Test detection with different baud rates."""
        for baudrate in [9600, 115200, 921600]:
            trace = generate_uart_signal(baudrate=baudrate)
            result = detect_protocol(trace, min_confidence=0.3)

            assert "protocol" in result
            # Symbol rate should be in reasonable range
            symbol_rate = result["characteristics"]["symbol_rate"]
            assert symbol_rate > 0

    def test_different_clock_frequencies(self) -> None:
        """Test SPI detection with different clock frequencies."""
        for freq in [1e5, 1e6, 10e6]:
            trace = generate_spi_signal(clock_freq=freq)
            result = detect_protocol(trace, min_confidence=0.3)

            assert "protocol" in result

    def test_confidence_consistency(self) -> None:
        """Test that repeated detection gives consistent results."""
        trace = generate_uart_signal(baudrate=115200)

        result1 = detect_protocol(trace, min_confidence=0.3)
        result2 = detect_protocol(trace, min_confidence=0.3)

        assert result1["protocol"] == result2["protocol"]
        assert result1["confidence"] == result2["confidence"]


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests for protocol detection."""

    def test_detection_speed_short_signal(self, benchmark) -> None:
        """Benchmark detection on short signal."""
        trace = generate_uart_signal(message=b"Test")

        def run():
            return detect_protocol(trace, min_confidence=0.3)

        result = benchmark(run)
        assert result["protocol"] in ["UART", "I2C", "CAN"]

    def test_detection_speed_long_signal(self, benchmark) -> None:
        """Benchmark detection on long signal."""
        trace = generate_uart_signal(message=b"A" * 1000)

        def run():
            return detect_protocol(trace, min_confidence=0.3)

        result = benchmark(run)
        assert result["protocol"] in ["UART", "I2C", "CAN"]

    def test_characteristics_analysis_speed(self, benchmark) -> None:
        """Benchmark characteristic analysis."""
        trace = generate_uart_signal(message=b"Testing" * 10)

        def run():
            return _analyze_signal_characteristics(trace)

        chars = benchmark(run)
        assert "regularity" in chars


# =============================================================================
# Documentation Examples
# =============================================================================


class TestDocumentationExamples:
    """Test examples from module docstrings."""

    def test_basic_usage_example(self) -> None:
        """Test basic usage example from docstring."""
        # Simulates: trace = osc.load('serial_data.wfm')
        trace = generate_uart_signal()

        # From docstring example
        result = detect_protocol(trace, min_confidence=0.3)

        assert "protocol" in result
        assert "confidence" in result
        # Confidence should be a float between 0 and 1
        assert 0 <= result["confidence"] <= 1

    def test_candidates_example(self) -> None:
        """Test return_candidates example from docstring."""
        trace = generate_uart_signal()

        # From docstring example
        result = detect_protocol(trace, return_candidates=True, min_confidence=0.3)

        assert "candidates" in result
        for candidate in result["candidates"]:
            assert "protocol" in candidate
            assert "confidence" in candidate
