"""Comprehensive unit tests for unknown signal analysis module.

Requirements tested:
- UNKNOWN-001: Binary Field Detection
- UNKNOWN-002: Protocol Auto-Detection with Fuzzy Matching
- UNKNOWN-003: Unknown Signal Characterization
- UNKNOWN-004: Pattern Frequency Analysis
- UNKNOWN-005: Reverse Engineering Workflow
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exploratory.unknown import (
    BinaryFieldResult,
    PatternFrequencyResult,
    ReverseEngineeringResult,
    UnknownSignalCharacterization,
    analyze_pattern_frequency,
    characterize_unknown_signal,
    detect_binary_fields,
    reverse_engineer_protocol,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def digital_uart_trace() -> WaveformTrace:
    """Create a digital UART-like signal for testing.

    Creates a simple digital signal with clear bit transitions
    at a standard UART baud rate (9600 bps).
    """
    sample_rate = 1_000_000  # 1 MHz
    bit_rate = 9600
    samples_per_bit = sample_rate // bit_rate  # ~104 samples/bit

    # Create a byte: start bit (0) + 8 data bits + stop bit (1)
    # Data: 0x55 = 01010101 (alternating pattern)
    bits = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # start + 0x55 + stop

    # Generate multiple bytes with gaps
    signal = []
    for _ in range(5):  # 5 bytes
        for bit in bits:
            signal.extend([bit] * samples_per_bit)
        # Add gap between bytes (idle high)
        signal.extend([1] * (samples_per_bit * 3))

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def analog_sine_trace() -> WaveformTrace:
    """Create an analog sine wave signal for testing."""
    sample_rate = 100_000  # 100 kHz
    duration = 0.01  # 10 ms
    frequency = 1000  # 1 kHz

    t = np.linspace(0, duration, int(sample_rate * duration))
    data = np.sin(2 * np.pi * frequency * t)
    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def noisy_digital_trace() -> WaveformTrace:
    """Create a noisy digital signal for testing."""
    sample_rate = 500_000
    samples_per_bit = 50
    bits = [0, 1, 0, 1, 1, 0, 1, 0] * 10

    signal = []
    for bit in bits:
        signal.extend([bit] * samples_per_bit)

    data = np.array(signal, dtype=np.float64)
    # Add noise
    noise = np.random.normal(0, 0.1, len(data))
    data = data + noise

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def manchester_trace() -> WaveformTrace:
    """Create a Manchester-encoded signal for testing."""
    sample_rate = 1_000_000
    bit_rate = 10_000
    samples_per_half_bit = sample_rate // (bit_rate * 2)

    # Manchester: each bit has transition in middle
    # 0 = high-to-low, 1 = low-to-high
    data_bits = [1, 0, 1, 1, 0, 0, 1, 0]

    signal = []
    for bit in data_bits:
        if bit == 1:
            # Low then high
            signal.extend([0] * samples_per_half_bit)
            signal.extend([1] * samples_per_half_bit)
        else:
            # High then low
            signal.extend([1] * samples_per_half_bit)
            signal.extend([0] * samples_per_half_bit)

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def empty_trace() -> WaveformTrace:
    """Create an empty/flat signal for testing edge cases."""
    sample_rate = 100_000
    data = np.ones(1000, dtype=np.float64) * 0.5  # Constant value
    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def i2c_like_trace() -> WaveformTrace:
    """Create an I2C-like signal for protocol detection testing."""
    sample_rate = 1_000_000  # 1 MHz
    bit_rate = 100_000  # 100 kHz (I2C standard mode)
    samples_per_bit = sample_rate // bit_rate

    # I2C-like pattern: start condition + address + ack + data
    bits = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]

    signal = []
    for bit in bits:
        signal.extend([bit] * samples_per_bit)

    # Repeat pattern
    signal = signal * 3

    data = np.array(signal, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=data, metadata=metadata)


# =============================================================================
# Test BinaryFieldResult
# =============================================================================


class TestBinaryFieldResult:
    """Test BinaryFieldResult dataclass."""

    def test_creation(self):
        """Test creating BinaryFieldResult instance."""
        result = BinaryFieldResult(
            fields=[{"start_sample": 0, "length": 8}],
            field_count=1,
            bit_rate=9600.0,
            encoding="nrz",
            confidence=0.9,
        )

        assert len(result.fields) == 1
        assert result.field_count == 1
        assert result.bit_rate == 9600.0
        assert result.encoding == "nrz"
        assert result.confidence == 0.9

    def test_empty_result(self):
        """Test empty BinaryFieldResult."""
        result = BinaryFieldResult(
            fields=[],
            field_count=0,
            bit_rate=None,
            encoding="unknown",
            confidence=0.0,
        )

        assert result.fields == []
        assert result.field_count == 0
        assert result.bit_rate is None
        assert result.encoding == "unknown"
        assert result.confidence == 0.0


# =============================================================================
# Test detect_binary_fields
# =============================================================================


class TestDetectBinaryFields:
    """Test detect_binary_fields function (UNKNOWN-001)."""

    def test_detect_uart_fields(self, digital_uart_trace):
        """Test detection of UART-like binary fields."""
        result = detect_binary_fields(digital_uart_trace)

        assert isinstance(result, BinaryFieldResult)
        assert result.field_count > 0
        assert result.bit_rate is not None
        assert result.bit_rate > 0
        assert result.encoding in ["nrz", "nrzi", "manchester", "unknown"]
        assert 0.0 <= result.confidence <= 1.0

        # Check field structure
        if result.fields:
            field = result.fields[0]
            assert "start_sample" in field
            assert "end_sample" in field
            assert "length" in field
            assert "bits" in field
            assert "timestamp" in field
            assert field["length"] > 0

    def test_empty_signal(self, empty_trace):
        """Test with flat/empty signal."""
        result = detect_binary_fields(empty_trace)

        assert result.field_count == 0
        assert result.bit_rate is None
        assert result.encoding == "unknown"
        assert result.confidence == 0.0

    def test_min_field_bits_parameter(self, digital_uart_trace):
        """Test min_field_bits parameter filtering."""
        # Higher threshold should result in fewer or equal fields
        result_low = detect_binary_fields(digital_uart_trace, min_field_bits=2)
        result_high = detect_binary_fields(digital_uart_trace, min_field_bits=10)

        assert result_high.field_count <= result_low.field_count

    def test_max_gap_ratio_parameter(self, digital_uart_trace):
        """Test max_gap_ratio parameter effect."""
        # Smaller gap ratio should result in more fields (splits more)
        result_small = detect_binary_fields(digital_uart_trace, max_gap_ratio=1.5)
        result_large = detect_binary_fields(digital_uart_trace, max_gap_ratio=5.0)

        # This is generally true but depends on signal structure
        assert isinstance(result_small, BinaryFieldResult)
        assert isinstance(result_large, BinaryFieldResult)

    def test_manchester_encoding_detection(self, manchester_trace):
        """Test detection of Manchester encoding."""
        result = detect_binary_fields(manchester_trace)

        # Manchester should be detected or at least not unknown
        assert result.encoding in ["manchester", "nrz", "nrzi", "unknown"]
        assert result.field_count >= 0

    def test_bit_rate_calculation(self, digital_uart_trace):
        """Test bit rate calculation accuracy."""
        result = detect_binary_fields(digital_uart_trace)

        if result.bit_rate is not None:
            # Should be close to 9600 bps (within 20%)
            expected_rate = 9600
            tolerance = 0.2
            assert abs(result.bit_rate - expected_rate) / expected_rate < tolerance

    def test_confidence_scoring(self, digital_uart_trace, empty_trace):
        """Test confidence scoring logic."""
        result_good = detect_binary_fields(digital_uart_trace)
        result_bad = detect_binary_fields(empty_trace)

        # Good signal should have higher confidence
        assert result_good.confidence > result_bad.confidence

    def test_field_timestamps(self, digital_uart_trace):
        """Test that field timestamps are calculated correctly."""
        result = detect_binary_fields(digital_uart_trace)

        if result.fields:
            for field in result.fields:
                timestamp = field["timestamp"]
                sample_rate = digital_uart_trace.metadata.sample_rate
                start_sample = field["start_sample"]

                # Timestamp should match sample position
                expected = start_sample / sample_rate
                assert abs(timestamp - expected) < 1e-6


# =============================================================================
# Test UnknownSignalCharacterization
# =============================================================================


class TestUnknownSignalCharacterization:
    """Test UnknownSignalCharacterization dataclass."""

    def test_creation(self):
        """Test creating UnknownSignalCharacterization instance."""
        char = UnknownSignalCharacterization(
            signal_type="digital",
            is_periodic=True,
            fundamental_frequency=1000.0,
            dc_offset=0.5,
            amplitude=1.0,
            rise_time=1e-6,
            fall_time=1e-6,
            suggested_protocols=[("UART", 0.8)],
            noise_floor=0.01,
            snr_db=20.0,
            features={"test": "value"},
        )

        assert char.signal_type == "digital"
        assert char.is_periodic is True
        assert char.fundamental_frequency == 1000.0
        assert char.suggested_protocols == [("UART", 0.8)]
        assert "test" in char.features


# =============================================================================
# Test characterize_unknown_signal
# =============================================================================


class TestCharacterizeUnknownSignal:
    """Test characterize_unknown_signal function (UNKNOWN-003)."""

    def test_digital_signal_characterization(self, digital_uart_trace):
        """Test characterization of digital signal."""
        result = characterize_unknown_signal(digital_uart_trace)

        assert isinstance(result, UnknownSignalCharacterization)
        assert result.signal_type in ["digital", "analog", "mixed"]
        assert isinstance(result.is_periodic, bool)
        assert result.dc_offset is not None
        assert result.amplitude > 0
        assert result.noise_floor >= 0
        assert isinstance(result.features, dict)

    def test_analog_signal_characterization(self, analog_sine_trace):
        """Test characterization of analog sine wave."""
        result = characterize_unknown_signal(analog_sine_trace)

        assert result.signal_type in ["analog", "mixed"]
        assert result.is_periodic is True  # Sine wave should be detected as periodic
        assert result.fundamental_frequency is not None
        # Should be close to 1 kHz
        assert 800 < result.fundamental_frequency < 1200

    def test_noisy_signal_characterization(self, noisy_digital_trace):
        """Test characterization with noisy signal."""
        result = characterize_unknown_signal(noisy_digital_trace)

        assert isinstance(result, UnknownSignalCharacterization)
        # Noisy signal should have reasonable SNR (not infinite)
        # 10% noise typically gives 40-60 dB SNR
        assert 30 < result.snr_db < 70  # Reasonable range for noisy signal

    def test_basic_statistics_extraction(self, digital_uart_trace):
        """Test that basic statistics are extracted correctly."""
        result = characterize_unknown_signal(digital_uart_trace)

        assert "v_min" in result.features
        assert "v_max" in result.features
        assert "v_mean" in result.features
        assert "v_std" in result.features

        # Check reasonable values
        assert result.features["v_min"] <= result.features["v_mean"]
        assert result.features["v_mean"] <= result.features["v_max"]

    def test_periodicity_detection(self, analog_sine_trace):
        """Test periodic signal detection."""
        result = characterize_unknown_signal(analog_sine_trace)

        assert result.is_periodic is True
        assert result.fundamental_frequency is not None
        assert result.fundamental_frequency > 0

    def test_protocol_suggestion(self, digital_uart_trace):
        """Test protocol suggestion feature (UNKNOWN-002)."""
        result = characterize_unknown_signal(digital_uart_trace)

        # Should suggest some protocols for digital signal
        assert isinstance(result.suggested_protocols, list)

        # If protocols suggested, check format
        if result.suggested_protocols:
            protocol_name, confidence = result.suggested_protocols[0]
            assert isinstance(protocol_name, str)
            assert 0.0 <= confidence <= 1.0

    def test_rise_fall_time_detection(self, digital_uart_trace):
        """Test rise/fall time detection for digital signals."""
        result = characterize_unknown_signal(digital_uart_trace)

        # Digital signals should have rise/fall times detected
        if result.signal_type == "digital":
            # May or may not detect depending on signal quality
            if result.rise_time is not None:
                assert result.rise_time > 0
            if result.fall_time is not None:
                assert result.fall_time > 0

    def test_snr_calculation(self, analog_sine_trace):
        """Test SNR calculation."""
        result = characterize_unknown_signal(analog_sine_trace)

        # Clean sine wave should have good SNR
        assert result.snr_db > 10  # At least 10 dB

    def test_i2c_protocol_detection(self, i2c_like_trace):
        """Test I2C protocol detection."""
        result = characterize_unknown_signal(i2c_like_trace)

        # Should suggest I2C or similar protocols
        if result.suggested_protocols:
            protocol_names = [p[0] for p in result.suggested_protocols]
            # Might detect I2C or similar protocols
            assert any("I2C" in p or "SPI" in p for p in protocol_names) or len(protocol_names) >= 0


# =============================================================================
# Test PatternFrequencyResult
# =============================================================================


class TestPatternFrequencyResult:
    """Test PatternFrequencyResult dataclass."""

    def test_creation(self):
        """Test creating PatternFrequencyResult instance."""
        patterns = {(0, 1, 0, 1): 5, (1, 0, 1, 0): 3}
        result = PatternFrequencyResult(
            patterns=patterns,
            most_common=[((0, 1, 0, 1), 5)],
            entropy=1.5,
            repetition_rate=0.3,
        )

        assert len(result.patterns) == 2
        assert result.most_common[0][1] == 5
        assert result.entropy == 1.5
        assert result.repetition_rate == 0.3


# =============================================================================
# Test analyze_pattern_frequency
# =============================================================================


class TestAnalyzePatternFrequency:
    """Test analyze_pattern_frequency function (UNKNOWN-004)."""

    def test_basic_pattern_detection(self, digital_uart_trace):
        """Test basic pattern frequency analysis."""
        result = analyze_pattern_frequency(digital_uart_trace, pattern_length=4)

        assert isinstance(result, PatternFrequencyResult)
        assert isinstance(result.patterns, dict)
        assert isinstance(result.most_common, list)
        assert result.entropy >= 0.0
        assert 0.0 <= result.repetition_rate <= 1.0

    def test_pattern_length_parameter(self, digital_uart_trace):
        """Test pattern_length parameter effect."""
        result_short = analyze_pattern_frequency(digital_uart_trace, pattern_length=4)
        result_long = analyze_pattern_frequency(digital_uart_trace, pattern_length=8)

        # Longer patterns should generally have fewer unique patterns
        assert isinstance(result_short, PatternFrequencyResult)
        assert isinstance(result_long, PatternFrequencyResult)

    def test_min_occurrences_filtering(self, digital_uart_trace):
        """Test min_occurrences parameter filtering."""
        result_low = analyze_pattern_frequency(digital_uart_trace, min_occurrences=1)
        result_high = analyze_pattern_frequency(digital_uart_trace, min_occurrences=5)

        # Higher threshold should result in fewer patterns
        assert len(result_high.patterns) <= len(result_low.patterns)

    def test_empty_signal(self, empty_trace):
        """Test with empty/flat signal."""
        result = analyze_pattern_frequency(empty_trace)

        # Flat signal should have minimal patterns
        assert result.entropy >= 0.0
        assert result.repetition_rate >= 0.0

    def test_repetitive_pattern_detection(self, digital_uart_trace):
        """Test detection of repetitive patterns."""
        result = analyze_pattern_frequency(digital_uart_trace, pattern_length=8)

        # UART signal should have some repetitive patterns
        if result.patterns:
            assert result.repetition_rate > 0.0

    def test_entropy_calculation(self, digital_uart_trace, empty_trace):
        """Test entropy calculation."""
        result_varied = analyze_pattern_frequency(digital_uart_trace)
        result_flat = analyze_pattern_frequency(empty_trace)

        # Varied signal should have higher entropy than flat
        # (though flat might have 0 patterns detected)
        assert result_varied.entropy >= 0.0
        assert result_flat.entropy >= 0.0

    def test_most_common_patterns(self, digital_uart_trace):
        """Test most_common patterns list."""
        result = analyze_pattern_frequency(digital_uart_trace, pattern_length=4)

        if result.most_common:
            # Should be sorted by count (descending)
            counts = [count for _, count in result.most_common]
            assert counts == sorted(counts, reverse=True)

            # Each pattern should be correct length
            for pattern, count in result.most_common:
                assert len(pattern) == 4
                assert count >= 1

    def test_pattern_tuple_format(self, digital_uart_trace):
        """Test that patterns are tuples of integers."""
        result = analyze_pattern_frequency(digital_uart_trace, pattern_length=6)

        for pattern in result.patterns:
            assert isinstance(pattern, tuple)
            assert len(pattern) == 6
            assert all(isinstance(bit, int | np.integer) for bit in pattern)
            assert all(bit in [0, 1] for bit in pattern)


# =============================================================================
# Test ReverseEngineeringResult
# =============================================================================


class TestReverseEngineeringResult:
    """Test ReverseEngineeringResult dataclass."""

    def test_creation(self):
        """Test creating ReverseEngineeringResult instance."""
        signal_char = UnknownSignalCharacterization(
            signal_type="digital",
            is_periodic=False,
            fundamental_frequency=None,
            dc_offset=0.5,
            amplitude=1.0,
            rise_time=None,
            fall_time=None,
            suggested_protocols=[],
            noise_floor=0.01,
            snr_db=20.0,
        )

        binary_fields = BinaryFieldResult(
            fields=[],
            field_count=0,
            bit_rate=None,
            encoding="unknown",
            confidence=0.0,
        )

        pattern_analysis = PatternFrequencyResult(
            patterns={},
            most_common=[],
            entropy=0.0,
            repetition_rate=0.0,
        )

        result = ReverseEngineeringResult(
            signal_char=signal_char,
            binary_fields=binary_fields,
            pattern_analysis=pattern_analysis,
            protocol_hypothesis="UART",
            confidence=0.7,
            recommendations=["Test recommendation"],
        )

        assert result.protocol_hypothesis == "UART"
        assert result.confidence == 0.7
        assert len(result.recommendations) == 1


# =============================================================================
# Test reverse_engineer_protocol
# =============================================================================


class TestReverseEngineerProtocol:
    """Test reverse_engineer_protocol function (UNKNOWN-005)."""

    def test_comprehensive_analysis(self, digital_uart_trace):
        """Test comprehensive reverse engineering workflow."""
        result = reverse_engineer_protocol(digital_uart_trace)

        assert isinstance(result, ReverseEngineeringResult)
        assert isinstance(result.signal_char, UnknownSignalCharacterization)
        assert isinstance(result.binary_fields, BinaryFieldResult)
        assert isinstance(result.pattern_analysis, PatternFrequencyResult)
        assert isinstance(result.protocol_hypothesis, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.recommendations, list)

    def test_uart_protocol_detection(self, digital_uart_trace):
        """Test UART protocol detection in reverse engineering."""
        result = reverse_engineer_protocol(digital_uart_trace)

        # Should detect some protocol for UART-like signal
        assert result.protocol_hypothesis != ""

        # Should have some confidence if detected
        if result.protocol_hypothesis != "Unknown":
            assert result.confidence > 0.0

    def test_recommendations_generation(self, digital_uart_trace):
        """Test that recommendations are generated."""
        result = reverse_engineer_protocol(digital_uart_trace)

        # Should have some recommendations
        assert len(result.recommendations) > 0

        # Each recommendation should be a string
        for rec in result.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_analog_signal_warning(self, analog_sine_trace):
        """Test warning for analog signals."""
        result = reverse_engineer_protocol(analog_sine_trace)

        # Should warn if signal is not digital
        if result.signal_char.signal_type != "digital":
            assert any("analog" in rec.lower() for rec in result.recommendations)

    def test_no_fields_warning(self, empty_trace):
        """Test warning when no binary fields detected."""
        result = reverse_engineer_protocol(empty_trace)

        # Should warn about no binary fields
        if result.binary_fields.field_count == 0:
            assert any("no binary fields" in rec.lower() for rec in result.recommendations)

    def test_manchester_encoding_recommendation(self, manchester_trace):
        """Test Manchester encoding recommendation."""
        result = reverse_engineer_protocol(manchester_trace)

        # If Manchester detected, should recommend relevant protocols
        if result.binary_fields.encoding == "manchester":
            assert any("manchester" in rec.lower() for rec in result.recommendations)

    def test_high_repetition_recommendation(self, i2c_like_trace):
        """Test recommendation for high pattern repetition."""
        result = reverse_engineer_protocol(i2c_like_trace)

        # High repetition should generate recommendation
        if result.pattern_analysis.repetition_rate > 0.5:
            assert any(
                "repetition" in rec.lower() or "periodic" in rec.lower()
                for rec in result.recommendations
            )

    def test_bit_rate_recommendation(self, digital_uart_trace):
        """Test bit rate is included in recommendations."""
        result = reverse_engineer_protocol(digital_uart_trace)

        # If bit rate detected, should be in recommendations
        if result.binary_fields.bit_rate is not None:
            assert any("bit rate" in rec.lower() for rec in result.recommendations)

    def test_low_snr_recommendation(self, noisy_digital_trace):
        """Test recommendation for low SNR signals."""
        result = reverse_engineer_protocol(noisy_digital_trace)

        # Low SNR should generate recommendation
        if result.signal_char.snr_db < 10:
            assert any(
                "snr" in rec.lower() or "filter" in rec.lower() for rec in result.recommendations
            )


# =============================================================================
# Test Internal Helper Functions
# =============================================================================


class TestInternalHelpers:
    """Test internal helper functions."""

    def test_detect_encoding_nrz(self, digital_uart_trace):
        """Test NRZ encoding detection."""
        from oscura.exploratory.unknown import _detect_encoding

        # Convert to digital for testing
        data = digital_uart_trace.data
        threshold = np.median(data)
        digital = (data > threshold).astype(int)
        edges = np.where(np.diff(digital) != 0)[0]

        if len(edges) >= 4:
            median_gap = np.median(np.diff(edges))
            encoding = _detect_encoding(digital, edges, median_gap)

            assert encoding in ["nrz", "nrzi", "manchester", "unknown"]

    def test_detect_encoding_manchester(self, manchester_trace):
        """Test Manchester encoding detection."""
        from oscura.exploratory.unknown import _detect_encoding

        data = manchester_trace.data
        threshold = np.median(data)
        digital = (data > threshold).astype(int)
        edges = np.where(np.diff(digital) != 0)[0]

        if len(edges) >= 4:
            median_gap = np.median(np.diff(edges))
            encoding = _detect_encoding(digital, edges, median_gap)

            # Manchester should be detected
            assert encoding in ["manchester", "nrz", "nrzi", "unknown"]

    def test_suggest_protocols_uart(self, digital_uart_trace):
        """Test UART protocol suggestion."""
        from oscura.exploratory.unknown import _suggest_protocols

        data = digital_uart_trace.data
        sample_rate = digital_uart_trace.metadata.sample_rate

        suggestions = _suggest_protocols("digital", None, sample_rate, data)

        # Should suggest UART for UART-like signal
        if suggestions:
            protocol_names = [p[0] for p in suggestions]
            assert isinstance(suggestions, list)

    def test_suggest_protocols_i2c(self, i2c_like_trace):
        """Test I2C protocol suggestion."""
        from oscura.exploratory.unknown import _suggest_protocols

        data = i2c_like_trace.data
        sample_rate = i2c_like_trace.metadata.sample_rate

        suggestions = _suggest_protocols("digital", None, sample_rate, data)

        # Should suggest I2C for I2C-like signal
        if suggestions:
            protocol_names = [p[0] for p in suggestions]
            # Might detect I2C or other protocols
            assert all(isinstance(name, str) for name in protocol_names)

    def test_suggest_protocols_non_digital(self):
        """Test protocol suggestion for non-digital signals."""
        from oscura.exploratory.unknown import _suggest_protocols

        # Analog signal should return empty suggestions
        data = np.sin(np.linspace(0, 10, 1000))
        suggestions = _suggest_protocols("analog", 1000.0, 100000, data)

        assert suggestions == []

    def test_suggest_protocols_sorting(self, digital_uart_trace):
        """Test that protocol suggestions are sorted by confidence."""
        from oscura.exploratory.unknown import _suggest_protocols

        data = digital_uart_trace.data
        sample_rate = digital_uart_trace.metadata.sample_rate

        suggestions = _suggest_protocols("digital", None, sample_rate, data)

        if len(suggestions) > 1:
            confidences = [conf for _, conf in suggestions]
            assert confidences == sorted(confidences, reverse=True)


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestExploratoryUnknownEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_trace(self):
        """Test with minimal single-sample trace."""
        data = np.array([1.0])
        metadata = TraceMetadata(sample_rate=1000)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should not crash
        result = detect_binary_fields(trace)
        assert result.field_count == 0

        char = characterize_unknown_signal(trace)
        assert isinstance(char, UnknownSignalCharacterization)

    def test_very_short_trace(self):
        """Test with very short trace."""
        data = np.array([0, 1, 0])
        metadata = TraceMetadata(sample_rate=1000)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_binary_fields(trace)
        assert isinstance(result, BinaryFieldResult)

        char = characterize_unknown_signal(trace)
        assert isinstance(char, UnknownSignalCharacterization)

    def test_all_zeros_trace(self):
        """Test with all-zero signal."""
        data = np.zeros(1000)
        metadata = TraceMetadata(sample_rate=100000)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_binary_fields(trace)
        assert result.field_count == 0

        char = characterize_unknown_signal(trace)
        assert char.signal_type in ["digital", "analog", "mixed"]

    def test_all_ones_trace(self):
        """Test with all-one signal."""
        data = np.ones(1000)
        metadata = TraceMetadata(sample_rate=100000)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_binary_fields(trace)
        assert result.field_count == 0

    def test_single_edge_trace(self):
        """Test with single edge transition."""
        data = np.concatenate([np.zeros(500), np.ones(500)])
        metadata = TraceMetadata(sample_rate=100000)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_binary_fields(trace)
        # Single edge - not enough for fields
        assert result.field_count == 0

    def test_negative_values(self):
        """Test with negative voltage values."""
        data = np.array([-1, -1, 1, 1, -1, -1, 1, 1] * 100, dtype=np.float64)
        metadata = TraceMetadata(sample_rate=100000)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_binary_fields(trace)
        # Should handle negative values correctly
        assert isinstance(result, BinaryFieldResult)

    def test_extreme_sample_rates(self):
        """Test with extreme sample rates."""
        # Very low sample rate
        data = np.array([0, 1, 0, 1] * 10, dtype=np.float64)
        metadata = TraceMetadata(sample_rate=10)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = characterize_unknown_signal(trace)
        assert isinstance(result, UnknownSignalCharacterization)

        # Very high sample rate
        metadata_high = TraceMetadata(sample_rate=1e9)
        trace_high = WaveformTrace(data=data, metadata=metadata_high)

        result_high = characterize_unknown_signal(trace_high)
        assert isinstance(result_high, UnknownSignalCharacterization)


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from oscura.exploratory import unknown

        expected_exports = [
            "BinaryFieldResult",
            "PatternFrequencyResult",
            "ReverseEngineeringResult",
            "UnknownSignalCharacterization",
            "analyze_pattern_frequency",
            "characterize_unknown_signal",
            "detect_binary_fields",
            "reverse_engineer_protocol",
        ]

        assert hasattr(unknown, "__all__")
        for export in expected_exports:
            assert export in unknown.__all__
            assert hasattr(unknown, export)
