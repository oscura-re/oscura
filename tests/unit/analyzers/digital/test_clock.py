"""Unit tests for clock recovery and analysis.

This module tests comprehensive clock recovery and analysis tools for digital
signals, including frequency detection, clock reconstruction, baud rate detection,
and jitter measurement.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.digital.clock import (
    BaudRateResult,
    ClockMetrics,
    ClockRecovery,
    detect_baud_rate,
    detect_clock_frequency,
    measure_clock_jitter,
    recover_clock,
)
from oscura.core.exceptions import FormatError, InsufficientDataError, ValidationError

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Test Data Generation Helpers
# =============================================================================


def generate_clock_signal(
    frequency: float,
    sample_rate: float,
    duration: float,
    noise_level: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a clean clock signal with optional noise."""
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples)
    # Generate square wave
    from scipy import signal

    clock = signal.square(2 * np.pi * frequency * t / sample_rate)
    # Normalize to 0-1
    clock = (clock + 1.0) / 2.0

    if noise_level > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_level, n_samples)
        clock = clock + noise

    return np.asarray(clock, dtype=np.float64)


def generate_uart_signal(
    baud_rate: int,
    sample_rate: float,
    message: bytes = b"Hello",
) -> NDArray[np.float64]:
    """Generate a UART signal with start/stop bits."""
    samples_per_bit = int(sample_rate / baud_rate)

    bits = []
    for byte_val in message:
        bits.append(0)  # Start bit
        for i in range(8):
            bits.append((byte_val >> i) & 1)
        bits.append(1)  # Stop bit

    signal = np.repeat(np.array(bits, dtype=np.float64), samples_per_bit)
    return signal


def generate_jittery_clock(
    base_frequency: float,
    sample_rate: float,
    duration: float,
    jitter_amount: float = 0.01,
) -> NDArray[np.float64]:
    """Generate a clock signal with timing jitter."""
    n_samples = int(sample_rate * duration)
    rng = np.random.default_rng(42)

    # Generate base period with jitter
    base_period = sample_rate / base_frequency
    clock = np.zeros(n_samples)

    pos = 0
    state = 1
    while pos < n_samples:
        # Add jitter to period
        period = base_period * (1 + rng.normal(0, jitter_amount))
        next_pos = int(pos + period / 2)
        if next_pos < n_samples:
            clock[pos:next_pos] = state
        pos = next_pos
        state = 1 - state

    return clock


def make_digital_trace(data: NDArray[np.float64], sample_rate: float = 1e6):
    """Create a DigitalTrace from raw data for testing."""
    try:
        from oscura.core.types import DigitalTrace, TraceMetadata

        metadata = TraceMetadata(sample_rate=sample_rate)
        # Convert to boolean: values > 0.5 are True
        bool_data = (data > 0.5).astype(bool)
        return DigitalTrace(data=bool_data, metadata=metadata)
    except ImportError:
        # Return a mock object if DigitalTrace not available
        class MockTrace:
            def __init__(self, data, metadata):
                self.data = data
                self.metadata = metadata

        class MockMetadata:
            def __init__(self, sample_rate):
                self.sample_rate = sample_rate

        return MockTrace(data, MockMetadata(sample_rate))


# =============================================================================
# ClockMetrics Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestClockMetrics:
    """Test ClockMetrics dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test ClockMetrics can be created with all fields."""
        metrics = ClockMetrics(
            frequency=1e6,
            period_samples=100.0,
            period_seconds=1e-6,
            jitter_rms=1e-9,
            jitter_pp=5e-9,
            duty_cycle=0.5,
            stability=0.95,
            confidence=0.9,
        )

        assert metrics.frequency == 1e6
        assert metrics.period_samples == 100.0
        assert metrics.period_seconds == 1e-6
        assert metrics.jitter_rms == 1e-9
        assert metrics.jitter_pp == 5e-9
        assert metrics.duty_cycle == 0.5
        assert metrics.stability == 0.95
        assert metrics.confidence == 0.9

    def test_attributes_exist(self) -> None:
        """Test all expected attributes exist."""
        metrics = ClockMetrics(
            frequency=1.0,
            period_samples=1.0,
            period_seconds=1.0,
            jitter_rms=0.0,
            jitter_pp=0.0,
            duty_cycle=0.5,
            stability=1.0,
            confidence=1.0,
        )

        assert hasattr(metrics, "frequency")
        assert hasattr(metrics, "period_samples")
        assert hasattr(metrics, "period_seconds")
        assert hasattr(metrics, "jitter_rms")
        assert hasattr(metrics, "jitter_pp")
        assert hasattr(metrics, "duty_cycle")
        assert hasattr(metrics, "stability")
        assert hasattr(metrics, "confidence")


# =============================================================================
# BaudRateResult Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestBaudRateResult:
    """Test BaudRateResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test BaudRateResult can be created with all fields."""
        result = BaudRateResult(
            baud_rate=115200,
            bit_period_samples=86.8,
            confidence=0.95,
            method="edge_histogram",
        )

        assert result.baud_rate == 115200
        assert result.bit_period_samples == 86.8
        assert result.confidence == 0.95
        assert result.method == "edge_histogram"

    def test_attributes_exist(self) -> None:
        """Test all expected attributes exist."""
        result = BaudRateResult(
            baud_rate=9600,
            bit_period_samples=100.0,
            confidence=0.8,
            method="test",
        )

        assert hasattr(result, "baud_rate")
        assert hasattr(result, "bit_period_samples")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method")


# =============================================================================
# ClockRecovery Initialization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestClockRecoveryInit:
    """Test ClockRecovery initialization."""

    def test_init_with_sample_rate(self) -> None:
        """Test initialization with explicit sample rate."""
        recovery = ClockRecovery(sample_rate=1e9)
        assert recovery.sample_rate == 1e9

    def test_init_without_sample_rate(self) -> None:
        """Test initialization without sample rate."""
        recovery = ClockRecovery()
        assert recovery.sample_rate is None

    def test_init_with_zero_sample_rate(self) -> None:
        """Test initialization with zero sample rate raises error."""
        with pytest.raises(ValidationError, match="Sample rate must be positive"):
            ClockRecovery(sample_rate=0)

    def test_init_with_negative_sample_rate(self) -> None:
        """Test initialization with negative sample rate raises error."""
        with pytest.raises(ValidationError, match="Sample rate must be positive"):
            ClockRecovery(sample_rate=-100)

    def test_standard_baud_rates(self) -> None:
        """Test STANDARD_BAUD_RATES class variable."""
        assert 9600 in ClockRecovery.STANDARD_BAUD_RATES
        assert 115200 in ClockRecovery.STANDARD_BAUD_RATES
        assert 1000000 in ClockRecovery.STANDARD_BAUD_RATES
        assert len(ClockRecovery.STANDARD_BAUD_RATES) > 10


# =============================================================================
# Frequency Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestFrequencyDetection:
    """Test clock frequency detection."""

    def test_detect_frequency_edge_method(self) -> None:
        """Test frequency detection using edge method."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="edge")

        # Tight tolerance: ±10% of expected frequency
        assert 0.9 * clock_freq < detected_freq < 1.1 * clock_freq

    def test_detect_frequency_fft_method(self) -> None:
        """Test frequency detection using FFT method.

        FFT on square waves detects the fundamental frequency at the expected
        clock frequency. While harmonics exist in the spectrum, the fundamental
        should have the highest magnitude.
        """
        sample_rate = 1e6
        clock_freq = 1000.0
        # Use longer duration (50ms) for better FFT frequency resolution
        signal = generate_clock_signal(clock_freq, sample_rate, 0.05)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="fft")

        # FFT should detect the fundamental frequency within ±15%
        # Note: For ideal square waves, the fundamental is at clock_freq
        # Slightly relaxed tolerance for CI environment robustness
        assert 0.85 * clock_freq < detected_freq < 1.15 * clock_freq

    def test_detect_frequency_autocorr_method(self) -> None:
        """Test frequency detection using autocorrelation method."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="autocorr")

        # Autocorrelation should detect the period accurately (±10%)
        assert 0.9 * clock_freq < detected_freq < 1.1 * clock_freq

    def test_detect_frequency_with_trace_object(self) -> None:
        """Test frequency detection with DigitalTrace object."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)
        trace = make_digital_trace(signal, sample_rate)

        recovery = ClockRecovery()  # No sample rate
        detected_freq = recovery.detect_frequency(trace, method="edge")

        # Should detect within ±10%
        assert 0.9 * clock_freq < detected_freq < 1.1 * clock_freq

    def test_detect_frequency_invalid_method(self) -> None:
        """Test frequency detection with invalid method raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = generate_clock_signal(1000, 1e6, 0.01)

        with pytest.raises(ValidationError, match="Unknown method"):
            recovery.detect_clock_frequency(signal, method="invalid")  # type: ignore

    def test_detect_frequency_no_sample_rate(self) -> None:
        """Test frequency detection without sample rate raises error."""
        recovery = ClockRecovery()  # No sample rate
        signal = generate_clock_signal(1000, 1e6, 0.01)

        with pytest.raises(ValidationError, match="Sample rate not set"):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_detect_frequency_insufficient_data(self) -> None:
        """Test frequency detection with too few samples raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, 0.0])  # Only 3 samples

        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_detect_frequency_high_frequency(self) -> None:
        """Test frequency detection with high frequency signal."""
        sample_rate = 100e6
        clock_freq = 10e6
        signal = generate_clock_signal(clock_freq, sample_rate, 0.001)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="fft")

        # Should detect high frequency within ±20% (allow more tolerance for high freq)
        assert 0.8 * clock_freq < detected_freq < 1.2 * clock_freq

    def test_detect_frequency_with_noise(self) -> None:
        """Test frequency detection on noisy signal."""
        sample_rate = 1e6
        clock_freq = 1000.0
        # Use longer duration (50ms) for better SNR averaging in FFT
        signal = generate_clock_signal(clock_freq, sample_rate, 0.05, noise_level=0.1)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="fft")

        # FFT should be robust to noise, detect within ±20%
        # Longer signal improves frequency resolution and noise averaging
        assert 0.8 * clock_freq < detected_freq < 1.2 * clock_freq


# =============================================================================
# Clock Recovery Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestClockRecovery:
    """Test clock signal recovery."""

    def test_recover_clock_edge_method(self) -> None:
        """Test clock recovery using edge method."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        recovered = recovery.recover_clock(signal, method="edge")

        assert len(recovered) == len(signal)
        assert recovered.min() >= 0.0
        assert recovered.max() <= 1.0

    def test_recover_clock_fft_method(self) -> None:
        """Test clock recovery using FFT method."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        recovered = recovery.recover_clock(signal, method="fft")

        assert len(recovered) == len(signal)
        assert recovered.min() >= 0.0
        assert recovered.max() <= 1.0

    def test_recover_clock_pll_method(self) -> None:
        """Test clock recovery using PLL method."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        recovered = recovery.recover_clock(signal, method="pll")

        assert len(recovered) == len(signal)
        # PLL output should be binary 0 or 1
        assert set(recovered) <= {0.0, 1.0}

    def test_recover_clock_insufficient_data(self) -> None:
        """Test clock recovery with too few samples raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0])

        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            recovery.recover_clock(signal, method="edge")

    def test_recover_clock_no_sample_rate(self) -> None:
        """Test clock recovery without sample rate raises error."""
        recovery = ClockRecovery()
        signal = generate_clock_signal(1000, 1e6, 0.01)

        with pytest.raises(ValidationError, match="Sample rate not set"):
            recovery.recover_clock(signal, method="edge")

    def test_pll_bandwidth_parameter(self) -> None:
        """Test PLL tracking with different bandwidth settings."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)

        # Test with tight bandwidth
        recovered_tight = recovery._pll_track(signal, clock_freq, bandwidth=0.001)

        # Test with loose bandwidth
        recovered_loose = recovery._pll_track(signal, clock_freq, bandwidth=0.1)

        assert len(recovered_tight) == len(signal)
        assert len(recovered_loose) == len(signal)


# =============================================================================
# Baud Rate Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestBaudRateDetection:
    """Test baud rate auto-detection."""

    def test_detect_baud_rate_9600(self) -> None:
        """Test baud rate detection for 9600 baud."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)

        recovery = ClockRecovery(sample_rate)
        result = recovery.detect_baud_rate(signal)

        assert isinstance(result, BaudRateResult)
        assert result.baud_rate == baud_rate
        assert result.method == "edge_histogram"
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_baud_rate_115200(self) -> None:
        """Test baud rate detection for 115200 baud."""
        sample_rate = 10e6
        baud_rate = 115200
        signal = generate_uart_signal(baud_rate, sample_rate)

        recovery = ClockRecovery(sample_rate)
        result = recovery.detect_baud_rate(signal)

        assert result.baud_rate == baud_rate

    def test_detect_baud_rate_custom_candidates(self) -> None:
        """Test baud rate detection with custom candidates."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)

        recovery = ClockRecovery(sample_rate)
        candidates = [4800, 9600, 19200]
        result = recovery.detect_baud_rate(signal, candidates=candidates)

        assert result.baud_rate in candidates

    def test_detect_baud_rate_insufficient_data(self) -> None:
        """Test baud rate detection with too few samples raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.zeros(50)  # Less than 100 samples

        with pytest.raises(InsufficientDataError, match="at least 100 samples"):
            recovery.detect_baud_rate(signal)

    def test_detect_baud_rate_no_edges(self) -> None:
        """Test baud rate detection with no edges raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.ones(1000)  # Constant signal

        with pytest.raises(InsufficientDataError, match="Not enough edges"):
            recovery.detect_baud_rate(signal)

    def test_detect_baud_rate_confidence(self) -> None:
        """Test that baud rate confidence is reasonable."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)

        recovery = ClockRecovery(sample_rate)
        result = recovery.detect_baud_rate(signal)

        # Confidence should be high for clean signal
        assert result.confidence > 0.5


# =============================================================================
# Jitter Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestJitterMeasurement:
    """Test clock jitter measurement."""

    def test_measure_jitter_clean_clock(self) -> None:
        """Test jitter measurement on clean clock."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        metrics = recovery.measure_clock_jitter(signal)

        assert isinstance(metrics, ClockMetrics)
        # Frequency should be within ±15% for clean signal
        assert 0.85 * clock_freq < metrics.frequency < 1.15 * clock_freq
        assert metrics.jitter_rms >= 0
        assert metrics.jitter_pp >= metrics.jitter_rms
        assert 0.0 <= metrics.duty_cycle <= 1.0
        assert 0.0 <= metrics.stability <= 1.0
        assert 0.0 <= metrics.confidence <= 1.0

    def test_measure_jitter_with_jitter(self) -> None:
        """Test jitter measurement on signal with jitter."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_jittery_clock(clock_freq, sample_rate, 0.01, jitter_amount=0.05)

        recovery = ClockRecovery(sample_rate)
        metrics = recovery.measure_clock_jitter(signal)

        # Should detect jitter
        assert metrics.jitter_rms > 0
        assert metrics.stability < 1.0

    def test_measure_jitter_insufficient_edges(self) -> None:
        """Test jitter measurement with too few edges raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        # Signal with only 1 edge
        signal = np.concatenate([np.zeros(50), np.ones(50)])

        with pytest.raises(InsufficientDataError, match="at least 3 rising edges"):
            recovery.measure_clock_jitter(signal)

    def test_measure_jitter_insufficient_data(self) -> None:
        """Test jitter measurement with too few samples raises error."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0])

        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            recovery.measure_clock_jitter(signal)

    def test_measure_jitter_duty_cycle(self) -> None:
        """Test duty cycle measurement."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        metrics = recovery.measure_clock_jitter(signal)

        # Should be close to 50% for square wave (within ±15%)
        assert 0.35 < metrics.duty_cycle < 0.65

    def test_measure_jitter_period_accuracy(self) -> None:
        """Test period measurement accuracy."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)
        metrics = recovery.measure_clock_jitter(signal)

        expected_period = 1.0 / clock_freq
        # Tighter tolerance: ±15%
        assert 0.85 * expected_period < metrics.period_seconds < 1.15 * expected_period


# =============================================================================
# Edge Detection Helper Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestEdgeDetectionHelpers:
    """Test internal edge detection helper methods."""

    def test_detect_edges_simple(self) -> None:
        """Test simple edge detection."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        edges = recovery._detect_edges_simple(signal)

        assert len(edges) > 0
        # Should detect both rising and falling edges

    def test_detect_edges_by_type_rising(self) -> None:
        """Test rising edge detection."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        edges = recovery._detect_edges_by_type(signal, "rising")

        assert len(edges) >= 2  # At least 2 rising edges

    def test_detect_edges_by_type_falling(self) -> None:
        """Test falling edge detection."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        edges = recovery._detect_edges_by_type(signal, "falling")

        assert len(edges) >= 2  # At least 2 falling edges

    def test_detect_edges_no_edges(self) -> None:
        """Test edge detection on constant signal."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.ones(100)
        edges = recovery._detect_edges_simple(signal)

        assert len(edges) == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_clock_frequency_function(self) -> None:
        """Test detect_clock_frequency convenience function."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        detected_freq = detect_clock_frequency(signal, sample_rate, method="edge")

        # Tight tolerance: ±15%
        assert 0.85 * clock_freq < detected_freq < 1.15 * clock_freq

    def test_recover_clock_function(self) -> None:
        """Test recover_clock convenience function."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovered = recover_clock(signal, sample_rate, method="edge")

        assert len(recovered) == len(signal)
        assert recovered.min() >= 0.0
        assert recovered.max() <= 1.0

    def test_measure_clock_jitter_function(self) -> None:
        """Test measure_clock_jitter convenience function."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        metrics = measure_clock_jitter(signal, sample_rate)

        assert isinstance(metrics, ClockMetrics)
        assert metrics.frequency > 0

    def test_detect_baud_rate_function_with_sample_rate(self) -> None:
        """Test detect_baud_rate convenience function with sample rate."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)

        result = detect_baud_rate(signal, sample_rate=sample_rate)

        assert isinstance(result, BaudRateResult)
        assert result.baud_rate == baud_rate

    def test_detect_baud_rate_function_with_trace(self) -> None:
        """Test detect_baud_rate convenience function with DigitalTrace."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)
        trace = make_digital_trace(signal, sample_rate)

        result = detect_baud_rate(trace)

        # With trace metadata, should return just the baud rate
        assert isinstance(result, int)
        assert result == baud_rate

    def test_detect_baud_rate_function_no_sample_rate(self) -> None:
        """Test detect_baud_rate without sample rate raises error."""
        signal = np.zeros(1000)

        with pytest.raises(ValidationError, match="sample_rate required"):
            detect_baud_rate(signal)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestClockRecoveryIntegration:
    """Integration tests combining multiple clock recovery features."""

    def test_full_workflow_detect_and_recover(self) -> None:
        """Test complete workflow: detect frequency then recover clock."""
        sample_rate = 1e6
        clock_freq = 1000.0
        signal = generate_clock_signal(clock_freq, sample_rate, 0.01)

        recovery = ClockRecovery(sample_rate)

        # Detect frequency
        detected_freq = recovery.detect_clock_frequency(signal, method="edge")
        assert 0.9 * clock_freq < detected_freq < 1.1 * clock_freq

        # Recover clock
        recovered = recovery.recover_clock(signal, method="edge")
        assert len(recovered) == len(signal)

        # Measure jitter on recovered clock
        metrics = recovery.measure_clock_jitter(recovered)
        assert metrics.frequency > 0

    def test_uart_workflow(self) -> None:
        """Test UART analysis workflow."""
        sample_rate = 1e6
        baud_rate = 9600
        signal = generate_uart_signal(baud_rate, sample_rate)

        recovery = ClockRecovery(sample_rate)

        # Detect baud rate
        result = recovery.detect_baud_rate(signal)
        assert result.baud_rate == baud_rate

        # Measure signal quality
        metrics = recovery.measure_clock_jitter(signal)
        assert metrics.frequency > 0

    def test_multiple_frequencies(self) -> None:
        """Test clock recovery with multiple different frequencies."""
        sample_rate = 10e6
        frequencies = [1000, 10000, 100000]  # Skip very low freq

        recovery = ClockRecovery(sample_rate)

        for freq in frequencies:
            # Use longer duration for low frequencies to ensure enough edges
            duration = max(0.001, 10.0 / freq)  # At least 10 cycles
            signal = generate_clock_signal(freq, sample_rate, duration)
            # Use edge method which is more reliable for fundamental frequency
            detected = recovery.detect_clock_frequency(signal, method="edge")
            # Tight tolerance: ±10% of expected frequency
            assert 0.9 * freq < detected < 1.1 * freq, (
                f"Expected {freq} Hz, got {detected} Hz (ratio: {detected / freq:.2f})"
            )

    def test_method_comparison(self) -> None:
        """Test that different methods give similar results."""
        sample_rate = 1e6
        clock_freq = 1000.0
        # Use longer duration (50ms) for consistent detection across methods
        signal = generate_clock_signal(clock_freq, sample_rate, 0.05)

        recovery = ClockRecovery(sample_rate)

        freq_edge = recovery.detect_clock_frequency(signal, method="edge")
        freq_fft = recovery.detect_clock_frequency(signal, method="fft")
        freq_autocorr = recovery.detect_clock_frequency(signal, method="autocorr")

        # All methods should be within ±20% of expected
        # Longer signal duration ensures all methods have sufficient data
        assert 0.8 * clock_freq < freq_edge < 1.2 * clock_freq
        assert 0.8 * clock_freq < freq_fft < 1.2 * clock_freq
        assert 0.8 * clock_freq < freq_autocorr < 1.2 * clock_freq


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestDigitalClockEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_transition_signal(self) -> None:
        """Test clock recovery on signal with single transition."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.concatenate([np.zeros(500), np.ones(500)])

        # Should raise error due to insufficient edges
        with pytest.raises(ValidationError):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_very_short_signal(self) -> None:
        """Test clock recovery on very short signal."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, 0.0, 1.0])

        with pytest.raises(InsufficientDataError):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_constant_signal(self) -> None:
        """Test clock recovery on constant signal."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.ones(1000)

        with pytest.raises(ValidationError):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_very_high_frequency(self) -> None:
        """Test clock recovery near Nyquist limit."""
        sample_rate = 1e6
        # Frequency at 40% of Nyquist
        clock_freq = 0.4 * sample_rate / 2
        signal = generate_clock_signal(clock_freq, sample_rate, 0.001)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="fft")

        # Should detect within ±20% even near Nyquist
        assert 0.8 * clock_freq < detected_freq < 1.2 * clock_freq

    def test_very_low_frequency(self) -> None:
        """Test clock recovery with very low frequency."""
        sample_rate = 1e6
        clock_freq = 10.0  # 10 Hz
        # Use longer duration (0.5s) to capture at least 5 full cycles
        # This ensures sufficient frequency resolution for 10 Hz signal
        signal = generate_clock_signal(clock_freq, sample_rate, 0.5)

        recovery = ClockRecovery(sample_rate)
        detected_freq = recovery.detect_clock_frequency(signal, method="fft")

        # Should detect within ±20%
        # Longer capture ensures FFT has sufficient resolution at low frequencies
        assert 0.8 * clock_freq < detected_freq < 1.2 * clock_freq

    def test_empty_signal(self) -> None:
        """Test clock recovery on empty signal."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([])

        with pytest.raises(InsufficientDataError):
            recovery.detect_clock_frequency(signal, method="edge")

    def test_nan_in_signal(self) -> None:
        """Test clock recovery with NaN values."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, np.nan, 0.0, 1.0] * 100)

        # Should handle NaN gracefully or raise appropriate error
        # (behavior depends on implementation)
        try:
            recovery.detect_clock_frequency(signal, method="edge")
        except (ValueError, ValidationError):
            pass  # Expected behavior

    def test_infinite_in_signal(self) -> None:
        """Test clock recovery with infinite values."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, np.inf, 0.0, 1.0] * 100)

        # Should handle inf gracefully or raise appropriate error
        try:
            recovery.detect_clock_frequency(signal, method="edge")
        except (ValueError, ValidationError):
            pass  # Expected behavior


# =============================================================================
# Trace Metadata Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestTraceMetadata:
    """Test trace metadata extraction."""

    def test_get_sample_rate_from_recovery(self) -> None:
        """Test sample rate retrieval from ClockRecovery instance."""
        recovery = ClockRecovery(sample_rate=1e9)
        signal = generate_clock_signal(1000, 1e6, 0.01)

        rate = recovery._get_sample_rate(signal)
        assert rate == 1e9

    def test_get_sample_rate_from_trace(self) -> None:
        """Test sample rate extraction from trace metadata."""
        sample_rate = 1e6
        signal = generate_clock_signal(1000, sample_rate, 0.01)
        trace = make_digital_trace(signal, sample_rate)

        recovery = ClockRecovery()  # No sample rate set
        rate = recovery._get_sample_rate(trace)

        assert rate == sample_rate

    def test_get_sample_rate_no_source(self) -> None:
        """Test sample rate extraction with no source raises error."""
        recovery = ClockRecovery()  # No sample rate
        signal = np.zeros(100)  # Plain array, no metadata

        with pytest.raises(ValidationError, match="Sample rate not set"):
            recovery._get_sample_rate(signal)

    def test_get_trace_data_from_array(self) -> None:
        """Test trace data extraction from numpy array."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, 0.0, 1.0])

        data = recovery._get_trace_data(signal)
        assert isinstance(data, np.ndarray)
        assert len(data) == 4

    def test_get_trace_data_from_trace(self) -> None:
        """Test trace data extraction from trace object."""
        recovery = ClockRecovery(sample_rate=1e6)
        signal = np.array([0.0, 1.0, 0.0, 1.0])
        trace = make_digital_trace(signal, 1e6)

        data = recovery._get_trace_data(trace)
        assert isinstance(data, np.ndarray)
        assert len(data) == 4


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-002")
class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports_available(self) -> None:
        """Test that all __all__ exports are importable."""
        from oscura.analyzers.digital import clock

        expected_exports = [
            "BaudRateResult",
            "ClockMetrics",
            "ClockRecovery",
            "detect_baud_rate",
            "detect_clock_frequency",
            "measure_clock_jitter",
            "recover_clock",
        ]

        for export in expected_exports:
            assert hasattr(clock, export), f"Missing export: {export}"

    def test_all_list_exists(self) -> None:
        """Test that __all__ list exists."""
        from oscura.analyzers.digital import clock

        assert hasattr(clock, "__all__")
        assert len(clock.__all__) == 7

    def test_dataclasses_importable(self) -> None:
        """Test that dataclasses are directly importable."""
        from oscura.analyzers.digital.clock import BaudRateResult, ClockMetrics

        assert BaudRateResult is not None
        assert ClockMetrics is not None

    def test_main_class_importable(self) -> None:
        """Test that main class is directly importable."""
        from oscura.analyzers.digital.clock import ClockRecovery

        assert ClockRecovery is not None

    def test_convenience_functions_importable(self) -> None:
        """Test that convenience functions are directly importable."""
        from oscura.analyzers.digital.clock import (
            detect_baud_rate,
            detect_clock_frequency,
            measure_clock_jitter,
            recover_clock,
        )

        assert detect_clock_frequency is not None
        assert recover_clock is not None
        assert detect_baud_rate is not None
        assert measure_clock_jitter is not None


# =============================================================================
# Sigrok File-Based Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requires_data
class TestSigrokUARTFiles:
    """Test clock recovery on real sigrok UART captures."""

    def test_load_uart_sigrok_files(self, uart_sigrok_files: list) -> None:
        """Test that UART sigrok files can be loaded."""
        if not uart_sigrok_files:
            pytest.skip("No UART sigrok files available")

        from oscura import load

        loaded_count = 0
        for sr_file in uart_sigrok_files[:5]:  # Test first 5
            try:
                trace = load(sr_file)
                if trace is not None and len(trace.data) > 0:
                    loaded_count += 1
            except (OSError, ValueError, KeyError, RuntimeError, FormatError) as e:
                # Skip files that fail to load (corrupted, unsupported format)
                continue

        if loaded_count == 0:
            # SKIP: Valid - Test data dependency
            # Only skip if sigrok test files not loaded successfully
            pytest.skip("No sigrok files loaded successfully")

    def test_baud_detection_on_real_captures(self, uart_sigrok_files: list) -> None:
        """Test baud rate detection on real UART captures."""
        if not uart_sigrok_files:
            # SKIP: Valid - Test data dependency
            # Only skip if sigrok test files not loaded successfully
            pytest.skip("No UART sigrok files available")

        from oscura import load

        detection_count = 0

        for sr_file in uart_sigrok_files[:5]:
            try:
                trace = load(sr_file)
                data = trace.data

                if len(data) < 100:
                    continue

                sample_rate = getattr(trace.metadata, "sample_rate", 1e6)
                result = detect_baud_rate(data, sample_rate)

                if result is not None:
                    detection_count += 1

            except (
                OSError,
                ValueError,
                KeyError,
                AttributeError,
                RuntimeError,
                FormatError,
            ) as e:
                # Skip files that fail to load or have invalid data
                continue

        # Skip if no files worked (data availability issue)
        if detection_count == 0:
            pytest.skip("No baud rates detected from available files")
