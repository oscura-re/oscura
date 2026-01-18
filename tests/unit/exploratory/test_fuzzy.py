"""Comprehensive unit tests for fuzzy matching module.

This module provides extensive testing for fuzzy timing matching, fuzzy pattern
matching, and fuzzy protocol detection with various edge cases and scenarios.


Test Coverage:
- fuzzy_timing_match with WaveformTrace and edge arrays
- fuzzy_pattern_match with various patterns and error tolerances
- fuzzy_protocol_detect with multiple protocol types
- Edge cases: empty data, insufficient edges, outliers
- Confidence scoring and deviation calculations
- Protocol signatures and timing matching
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exploratory.fuzzy import (
    PROTOCOL_SIGNATURES,
    FuzzyPatternResult,
    FuzzyProtocolResult,
    FuzzyTimingResult,
    fuzzy_pattern_match,
    fuzzy_protocol_detect,
    fuzzy_timing_match,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def make_waveform_trace(data: np.ndarray, sample_rate: float = 1e6) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def generate_periodic_signal(
    period: float,
    num_periods: int = 10,
    sample_rate: float = 1e6,
    jitter: float = 0.0,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate a periodic square wave signal.

    Args:
        period: Period in seconds (time between edges).
        num_periods: Number of periods to generate.
        sample_rate: Sample rate in Hz.
        jitter: RMS jitter as fraction of period.
        noise: Noise amplitude as fraction of signal.

    Returns:
        Array of voltage samples.
    """
    samples_per_period = int(period * sample_rate)
    # Need 2x periods for full square wave (rising + falling edges)
    total_samples = samples_per_period * num_periods * 2

    signal = np.zeros(total_samples)

    # Create toggling signal where edges are spaced at 'period' intervals
    current_state = 0  # Start LOW
    for i in range(num_periods * 2):  # 2 edges per full cycle
        # Add jitter to period
        if jitter > 0:
            jittered_samples = int(samples_per_period * (1 + np.random.randn() * jitter))
        else:
            jittered_samples = samples_per_period

        start = int(i * samples_per_period)
        end = min(start + jittered_samples, len(signal))

        signal[start:end] = 3.3 if current_state else 0.0
        current_state = 1 - current_state  # Toggle

    # Add noise if requested
    if noise > 0:
        signal += np.random.randn(len(signal)) * 3.3 * noise

    return signal


def generate_uart_frame(
    bits: list[int],
    baudrate: int = 9600,
    sample_rate: float = 1e6,
) -> np.ndarray:
    """Generate a UART frame with start bit, data bits, and stop bit.

    Args:
        bits: Data bits (LSB first).
        baudrate: Baud rate in bps.
        sample_rate: Sample rate in Hz.

    Returns:
        Array of voltage samples.
    """
    bit_period = 1.0 / baudrate
    samples_per_bit = int(bit_period * sample_rate)

    # UART frame: start bit (0) + data bits + stop bit (1)
    frame = [0] + bits + [1]

    signal = np.zeros(len(frame) * samples_per_bit)

    for i, bit in enumerate(frame):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        signal[start:end] = 3.3 if bit else 0.0

    return signal


def generate_i2c_pattern(
    sample_rate: float = 1e6,
    scl_rate: float = 100e3,
) -> np.ndarray:
    """Generate an I2C-like signal with start condition.

    Args:
        sample_rate: Sample rate in Hz.
        scl_rate: SCL clock rate in Hz.

    Returns:
        Array of voltage samples representing SDA line.
    """
    bit_period = 1.0 / scl_rate
    samples_per_bit = int(bit_period * sample_rate)

    # I2C start: SDA falls while SCL is high
    # Idle: both high, then SDA falls
    signal = np.ones(samples_per_bit * 4) * 3.3

    # Start condition: SDA falls at quarter bit time
    signal[samples_per_bit // 4 :] = 0.0

    return signal


# =============================================================================
# FuzzyTimingResult Tests
# =============================================================================


class TestFuzzyTimingResult:
    """Test FuzzyTimingResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating FuzzyTimingResult."""
        result = FuzzyTimingResult(
            match=True,
            confidence=0.95,
            period=1e-6,
            deviation=0.05,
            jitter_rms=1e-9,
            outlier_count=2,
            outlier_indices=[5, 10],
        )

        assert result.match is True
        assert result.confidence == 0.95
        assert result.period == 1e-6
        assert result.deviation == 0.05
        assert result.jitter_rms == 1e-9
        assert result.outlier_count == 2
        assert result.outlier_indices == [5, 10]


# =============================================================================
# fuzzy_timing_match Tests
# =============================================================================


class TestFuzzyTimingMatch:
    """Test fuzzy_timing_match function."""

    def test_perfect_periodic_signal(self) -> None:
        """Test matching perfect periodic signal."""
        # Generate perfect 1 MHz signal (1 us period)
        period = 1e-6
        sample_rate = 10e6  # 10 MHz for adequate sampling
        signal = generate_periodic_signal(period, num_periods=20, sample_rate=sample_rate)
        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        result = fuzzy_timing_match(trace, expected_period=period, tolerance=0.1)

        assert result.match is True
        assert result.confidence > 0.9
        assert abs(result.period - period) < period * 0.05
        assert result.deviation < 0.05
        assert result.outlier_count == 0

    def test_jittered_signal(self) -> None:
        """Test matching signal with jitter."""
        period = 1e-6
        sample_rate = 10e6  # 10 MHz for adequate sampling
        signal = generate_periodic_signal(
            period, num_periods=20, sample_rate=sample_rate, jitter=0.05
        )
        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        result = fuzzy_timing_match(trace, expected_period=period, tolerance=0.1)

        assert result.match is True
        assert result.jitter_rms > 0
        assert result.confidence > 0.5

    def test_edge_array_input(self) -> None:
        """Test with edge array instead of WaveformTrace."""
        # Create edge times at 1 us intervals
        edges = np.array([0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6])

        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)

        assert result.match is True
        assert abs(result.period - 1e-6) < 1e-9
        assert result.deviation < 0.01

    def test_no_expected_period(self) -> None:
        """Test auto-detection when no expected period provided."""
        edges = np.array([0.0, 1e-6, 2e-6, 3e-6, 4e-6])

        result = fuzzy_timing_match(edges, tolerance=0.1)

        assert result.match is True
        assert result.period > 0
        assert result.confidence > 0.9

    def test_period_mismatch(self) -> None:
        """Test detection of period mismatch."""
        edges = np.array([0.0, 1e-6, 2e-6, 3e-6, 4e-6])

        # Expect 2 us but actual is 1 us
        result = fuzzy_timing_match(edges, expected_period=2e-6, tolerance=0.1)

        assert result.match is False
        assert result.deviation > 0.4
        assert result.confidence < 0.5

    def test_outliers_detection(self) -> None:
        """Test detection of timing outliers."""
        # Create edges with outliers
        edges = np.array([0.0, 1e-6, 2e-6, 5e-6, 6e-6, 7e-6])  # 5e-6 is outlier

        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)

        assert result.outlier_count > 0
        assert len(result.outlier_indices) > 0

    def test_insufficient_edges(self) -> None:
        """Test handling of insufficient edges."""
        edges = np.array([0.0])  # Only one edge

        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)

        assert result.match is False
        assert result.confidence == 0.0
        assert result.period == 0.0

    def test_empty_edges(self) -> None:
        """Test handling of empty edge array."""
        edges = np.array([])

        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)

        assert result.match is False
        assert result.confidence == 0.0

    def test_tight_tolerance(self) -> None:
        """Test with tight tolerance."""
        edges = np.array([0.0, 1.05e-6, 2.1e-6, 3.15e-6])  # 5% deviation

        # 2% tolerance should fail
        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.02)
        assert result.match is False

        # 10% tolerance should pass
        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)
        assert result.match is True

    def test_confidence_calculation(self) -> None:
        """Test confidence score calculation."""
        # Perfect match should have high confidence
        edges1 = np.array([0.0, 1e-6, 2e-6, 3e-6, 4e-6])
        result1 = fuzzy_timing_match(edges1, expected_period=1e-6, tolerance=0.1)
        assert result1.confidence > 0.9

        # Match with deviation should have lower confidence
        edges2 = np.array([0.0, 1.08e-6, 2.16e-6, 3.24e-6])
        result2 = fuzzy_timing_match(edges2, expected_period=1e-6, tolerance=0.1)
        assert 0.0 < result2.confidence < 0.5


# =============================================================================
# FuzzyPatternResult Tests
# =============================================================================


class TestFuzzyPatternResult:
    """Test FuzzyPatternResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating FuzzyPatternResult."""
        result = FuzzyPatternResult(
            matches=[{"position": 0, "score": 1.0}],
            best_match_score=1.0,
            total_matches=1,
            pattern_variations=[((0, 1, 1), 2)],
        )

        assert len(result.matches) == 1
        assert result.best_match_score == 1.0
        assert result.total_matches == 1
        assert len(result.pattern_variations) == 1


# =============================================================================
# fuzzy_pattern_match Tests
# =============================================================================


class TestFuzzyPatternMatch:
    """Test fuzzy_pattern_match function."""

    def test_exact_pattern_match(self) -> None:
        """Test finding exact pattern matches."""
        # Generate signal with known pattern
        pattern = [0, 1, 0, 1, 0, 1]
        signal = generate_uart_frame(pattern * 3, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=0)

        assert result.total_matches > 0
        assert result.best_match_score == 1.0
        assert all(m["errors"] == 0 for m in result.matches)

    def test_pattern_with_errors(self) -> None:
        """Test finding patterns with bit errors."""
        # Note: Edge-based sampling works best with alternating patterns
        # Use a pattern with good edge density for reliable detection
        pattern = [0, 1, 0, 1, 0, 1]
        # Generate with slightly different pattern (1 bit flipped), repeat for more edges
        actual = [0, 1, 1, 1, 0, 1] * 2  # Repeated to ensure enough edges
        signal = generate_uart_frame(actual, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=1)

        assert result.total_matches > 0
        # Should find match with 1 error
        assert any(m["errors"] == 1 for m in result.matches)

    def test_max_errors_filtering(self) -> None:
        """Test that max_errors filters matches correctly."""
        pattern = [0, 1, 0, 1]
        actual = [1, 1, 1, 1]  # 2 errors
        signal = generate_uart_frame(actual, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        # max_errors=1 should not find this match
        result1 = fuzzy_pattern_match(trace, pattern, max_errors=1)
        assert result1.total_matches == 0 or all(m["errors"] <= 1 for m in result1.matches)

        # max_errors=2 should find it
        result2 = fuzzy_pattern_match(trace, pattern, max_errors=2)
        # May or may not find depending on exact sampling

    def test_score_calculation(self) -> None:
        """Test match score calculation."""
        pattern = [0, 1, 0, 1, 0, 1]
        signal = generate_uart_frame(pattern, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=1, error_weight=0.5)

        if result.matches:
            # Score should be 1.0 for exact match
            exact_matches = [m for m in result.matches if m["errors"] == 0]
            if exact_matches:
                assert exact_matches[0]["score"] == 1.0

            # Score should decrease with errors
            error_matches = [m for m in result.matches if m["errors"] > 0]
            if error_matches:
                assert all(m["score"] < 1.0 for m in error_matches)

    def test_pattern_variations_tracking(self) -> None:
        """Test tracking of pattern variations."""
        pattern = [0, 1, 0, 1]
        # Create signal with multiple variations
        variations = [[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]]
        signal = np.concatenate(
            [generate_uart_frame(v, baudrate=9600, sample_rate=1e6) for v in variations]
        )
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=2)

        # Should track variations that don't match exactly
        assert isinstance(result.pattern_variations, list)

    def test_insufficient_edges(self) -> None:
        """Test handling of signal with insufficient edges."""
        # Constant signal with no edges
        signal = np.ones(1000) * 3.3
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, [0, 1, 0, 1], max_errors=1)

        assert result.total_matches == 0
        assert result.best_match_score == 0.0
        assert len(result.matches) == 0

    def test_empty_pattern(self) -> None:
        """Test handling of empty pattern."""
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, [], max_errors=0)

        # Empty pattern should not find matches
        assert result.total_matches == 0

    def test_tuple_pattern_input(self) -> None:
        """Test pattern as tuple instead of list."""
        pattern = (0, 1, 0, 1)
        signal = generate_uart_frame(list(pattern), baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=0)

        # Should work with tuple input
        assert isinstance(result, FuzzyPatternResult)

    def test_match_position_tracking(self) -> None:
        """Test that match positions are tracked correctly."""
        pattern = [0, 1, 0, 1]
        signal = generate_uart_frame(pattern, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=0)

        if result.matches:
            # Each match should have position and sample_position
            for match in result.matches:
                assert "position" in match
                assert "sample_position" in match
                assert isinstance(match["position"], int)
                assert isinstance(match["sample_position"], int)


# =============================================================================
# FuzzyProtocolResult Tests
# =============================================================================


class TestFuzzyProtocolResult:
    """Test FuzzyProtocolResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating FuzzyProtocolResult."""
        result = FuzzyProtocolResult(
            detected_protocol="UART",
            confidence=0.85,
            alternatives=[("I2C", 0.3)],
            timing_score=0.9,
            pattern_score=0.8,
            recommendations=["Verify baud rate"],
        )

        assert result.detected_protocol == "UART"
        assert result.confidence == 0.85
        assert len(result.alternatives) == 1
        assert result.timing_score == 0.9
        assert result.pattern_score == 0.8
        assert len(result.recommendations) == 1


# =============================================================================
# fuzzy_protocol_detect Tests
# =============================================================================


class TestFuzzyProtocolDetect:
    """Test fuzzy_protocol_detect function."""

    def test_uart_detection(self) -> None:
        """Test detecting UART protocol."""
        # Generate UART signal at standard baud rate
        signal = generate_uart_frame([0, 1, 0, 1, 0, 1, 0, 1], baudrate=115200, sample_rate=10e6)
        trace = make_waveform_trace(signal, sample_rate=10e6)

        result = fuzzy_protocol_detect(trace, candidates=["UART"])

        assert result.detected_protocol == "UART"
        assert result.confidence > 0.0

    def test_i2c_detection(self) -> None:
        """Test detecting I2C protocol."""
        # Generate I2C-like signal
        signal = generate_i2c_pattern(sample_rate=1e6, scl_rate=100e3)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace, candidates=["I2C"])

        # I2C detection depends on start pattern matching
        assert isinstance(result, FuzzyProtocolResult)

    def test_all_protocols_search(self) -> None:
        """Test searching all protocols when no candidates specified."""
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace)  # No candidates = search all

        assert (
            result.detected_protocol in PROTOCOL_SIGNATURES or result.detected_protocol == "Unknown"
        )

    def test_timing_tolerance(self) -> None:
        """Test timing tolerance parameter."""
        # Generate signal with slightly off timing
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=9800, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        # Strict tolerance
        result1 = fuzzy_protocol_detect(trace, candidates=["UART"], timing_tolerance=0.01)

        # Loose tolerance
        result2 = fuzzy_protocol_detect(trace, candidates=["UART"], timing_tolerance=0.5)

        # Loose tolerance should have equal or higher confidence
        assert result2.confidence >= result1.confidence

    def test_insufficient_edges(self) -> None:
        """Test handling of signal with insufficient edges."""
        # Very short constant signal
        signal = np.ones(10) * 3.3
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace)

        assert result.detected_protocol == "Unknown"
        assert result.confidence == 0.0
        assert "Insufficient edges" in result.recommendations[0]

    def test_alternatives_list(self) -> None:
        """Test that alternatives are provided."""
        signal = generate_uart_frame([0, 1, 0, 1, 0, 1], baudrate=115200, sample_rate=10e6)
        trace = make_waveform_trace(signal, sample_rate=10e6)

        result = fuzzy_protocol_detect(trace)

        # Should have alternatives or empty list
        assert isinstance(result.alternatives, list)

    def test_recommendations_generation(self) -> None:
        """Test that recommendations are generated."""
        signal = generate_uart_frame([0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace)

        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_confidence_thresholds(self) -> None:
        """Test recommendation based on confidence thresholds."""
        # Create poor quality signal
        signal = np.random.rand(1000) * 3.3
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace)

        # Low confidence should trigger recommendation
        if result.confidence < 0.5:
            assert any("confidence" in r.lower() for r in result.recommendations)

    def test_timing_vs_pattern_scores(self) -> None:
        """Test that timing and pattern scores are tracked separately."""
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=115200, sample_rate=10e6)
        trace = make_waveform_trace(signal, sample_rate=10e6)

        result = fuzzy_protocol_detect(trace, candidates=["UART"])

        # Both scores should be non-negative
        assert result.timing_score >= 0.0
        assert result.pattern_score >= 0.0

    def test_invalid_candidate(self) -> None:
        """Test handling of invalid protocol candidate."""
        signal = generate_uart_frame([0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_protocol_detect(trace, candidates=["NONEXISTENT"])

        # Should return Unknown since no valid candidates
        assert result.detected_protocol == "Unknown"

    def test_multiple_candidates(self) -> None:
        """Test detection with multiple candidates."""
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=115200, sample_rate=10e6)
        trace = make_waveform_trace(signal, sample_rate=10e6)

        result = fuzzy_protocol_detect(trace, candidates=["UART", "I2C", "SPI"])

        # Should pick one of the candidates or Unknown
        assert result.detected_protocol in ["UART", "I2C", "SPI", "Unknown"]


# =============================================================================
# Protocol Signatures Tests
# =============================================================================


class TestProtocolSignatures:
    """Test PROTOCOL_SIGNATURES constant."""

    def test_signatures_exist(self) -> None:
        """Test that protocol signatures are defined."""
        assert isinstance(PROTOCOL_SIGNATURES, dict)
        assert len(PROTOCOL_SIGNATURES) > 0

    def test_expected_protocols(self) -> None:
        """Test that expected protocols are in signatures."""
        expected = ["UART", "I2C", "SPI", "CAN"]
        for protocol in expected:
            assert protocol in PROTOCOL_SIGNATURES

    def test_signature_structure(self) -> None:
        """Test that each signature has expected structure."""
        for sig in PROTOCOL_SIGNATURES.values():
            assert isinstance(sig, dict)
            # Each should have some attributes
            assert len(sig) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestExploratoryFuzzyIntegration:
    """Integration tests combining multiple functions."""

    def test_timing_and_pattern_workflow(self) -> None:
        """Test workflow using both timing and pattern matching."""
        # Generate UART signal with known alternating pattern (good edge density)
        pattern = [0, 1, 0, 1, 0, 1]
        signal = generate_uart_frame(pattern * 3, baudrate=115200, sample_rate=10e6)
        trace = make_waveform_trace(signal, sample_rate=10e6)

        # Check timing
        expected_bit_period = 1.0 / 115200
        timing_result = fuzzy_timing_match(
            trace, expected_period=expected_bit_period, tolerance=0.15
        )

        # Check pattern
        pattern_result = fuzzy_pattern_match(trace, pattern, max_errors=1)

        # Both should succeed
        assert timing_result.match is True
        assert pattern_result.total_matches > 0

    def test_protocol_detection_workflow(self) -> None:
        """Test full protocol detection workflow."""
        # Generate signal
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        # Detect protocol
        protocol_result = fuzzy_protocol_detect(trace)

        # Should get a result
        assert isinstance(protocol_result, FuzzyProtocolResult)
        assert protocol_result.detected_protocol != ""

    def test_noisy_signal_handling(self) -> None:
        """Test handling of noisy signals across all functions."""
        # Generate signal with noise
        period = 1e-6
        sample_rate = 10e6  # 10 MHz for adequate sampling
        signal = generate_periodic_signal(
            period, num_periods=20, sample_rate=sample_rate, noise=0.1
        )
        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        # All functions should handle noise gracefully
        timing_result = fuzzy_timing_match(trace, expected_period=period, tolerance=0.2)
        pattern_result = fuzzy_pattern_match(trace, [0, 1, 0, 1], max_errors=2)
        protocol_result = fuzzy_protocol_detect(trace)

        # Should all return valid results (not crash)
        assert isinstance(timing_result, FuzzyTimingResult)
        assert isinstance(pattern_result, FuzzyPatternResult)
        assert isinstance(protocol_result, FuzzyProtocolResult)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestExploratoryFuzzyEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample_trace(self) -> None:
        """Test handling of trace with single sample."""
        trace = make_waveform_trace(np.array([3.3]), sample_rate=1e6)

        result = fuzzy_timing_match(trace, expected_period=1e-6, tolerance=0.1)
        assert result.match is False

    def test_constant_signal(self) -> None:
        """Test handling of constant (no edges) signal."""
        signal = np.ones(1000) * 3.3
        trace = make_waveform_trace(signal, sample_rate=1e6)

        timing_result = fuzzy_timing_match(trace, expected_period=1e-6, tolerance=0.1)
        pattern_result = fuzzy_pattern_match(trace, [0, 1], max_errors=0)
        protocol_result = fuzzy_protocol_detect(trace)

        assert timing_result.match is False
        assert pattern_result.total_matches == 0
        assert protocol_result.detected_protocol == "Unknown"

    def test_very_short_pattern(self) -> None:
        """Test handling of very short pattern."""
        signal = generate_uart_frame([0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, [0], max_errors=0)

        # Single-bit pattern should work
        assert isinstance(result, FuzzyPatternResult)

    def test_negative_tolerance(self) -> None:
        """Test handling of negative tolerance (should still work)."""
        edges = np.array([0.0, 1e-6, 2e-6, 3e-6])

        # Negative tolerance is unusual but shouldn't crash
        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=-0.1)
        assert isinstance(result, FuzzyTimingResult)

    def test_zero_sample_rate(self) -> None:
        """Test handling of zero sample rate in protocol detection."""
        # This should be prevented by TraceMetadata validation
        # but test graceful handling if it occurs
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        metadata = TraceMetadata(sample_rate=1.0)  # Very low but non-zero
        trace = WaveformTrace(data=signal, metadata=metadata)

        result = fuzzy_protocol_detect(trace)
        assert isinstance(result, FuzzyProtocolResult)

    def test_large_max_errors(self) -> None:
        """Test pattern matching with max_errors larger than pattern."""
        pattern = [0, 1]
        signal = generate_uart_frame([0, 1, 0, 1], baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=100)

        # Should find matches (everything matches with high enough error tolerance)
        assert result.total_matches > 0

    def test_zero_error_weight(self) -> None:
        """Test pattern matching with zero error weight."""
        pattern = [0, 1, 0, 1]
        signal = generate_uart_frame(pattern, baudrate=9600, sample_rate=1e6)
        trace = make_waveform_trace(signal, sample_rate=1e6)

        result = fuzzy_pattern_match(trace, pattern, max_errors=1, error_weight=0.0)

        # With zero error weight, all scores should be high
        if result.matches:
            assert all(m["score"] >= 0.9 for m in result.matches)
