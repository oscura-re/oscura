"""Tests for CAN message analysis algorithms."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import pytest

from oscura.automotive.can.analysis import MessageAnalyzer


class TestMessageAnalyzer:
    """Tests for MessageAnalyzer class."""

    def test_calculate_entropy(self):
        """Test Shannon entropy calculation."""
        # All same value = 0 entropy
        values = [0x12] * 100
        entropy = MessageAnalyzer.calculate_entropy(values)
        assert entropy == pytest.approx(0.0)

        # Two values equally distributed = 1 bit entropy
        values = [0x00] * 50 + [0xFF] * 50
        entropy = MessageAnalyzer.calculate_entropy(values)
        assert entropy == pytest.approx(1.0, rel=0.01)

        # All different values = high entropy
        values = list(range(256))
        entropy = MessageAnalyzer.calculate_entropy(values)
        assert entropy > 7.0  # Should be close to 8 bits

    def test_detect_counter_simple(self):
        """Test counter detection with simple incrementing sequence."""
        values = list(range(100))  # 0, 1, 2, ..., 99
        counter = MessageAnalyzer.detect_counter(values)

        assert counter is not None
        assert counter.increment == 1
        assert counter.pattern_type == "counter"
        assert counter.confidence > 0.9

    def test_detect_counter_with_wrap(self):
        """Test counter detection with wraparound."""
        values = list(range(250, 256)) + list(range(10))  # 250..255, 0..9
        counter = MessageAnalyzer.detect_counter(values)

        assert counter is not None
        assert counter.increment == 1
        assert counter.wraps_at == 255

    def test_detect_counter_increment_2(self):
        """Test counter detection with increment of 2."""
        values = list(range(0, 100, 2))  # 0, 2, 4, ..., 98
        counter = MessageAnalyzer.detect_counter(values)

        assert counter is not None
        assert counter.increment == 2
        assert counter.confidence > 0.9

    def test_detect_counter_no_pattern(self):
        """Test counter detection with no pattern (random data)."""
        values = [17, 42, 3, 99, 12, 88, 5, 63, 21, 77]
        counter = MessageAnalyzer.detect_counter(values)

        assert counter is None  # Should not detect counter in random data

    def test_analyze_byte(self, sample_can_messages):
        """Test byte-level analysis."""

        # Filter to message 0x280
        filtered = sample_can_messages.filter_by_id(0x280)

        # Analyze byte 0 (constant 0xAA)
        ba = MessageAnalyzer.analyze_byte(filtered, 0)
        assert ba.is_constant
        assert ba.most_common_value == 0xAA
        assert ba.entropy == pytest.approx(0.0)

        # Analyze byte 2 (RPM high byte - should vary)
        ba = MessageAnalyzer.analyze_byte(filtered, 2)
        assert not ba.is_constant
        assert ba.entropy > 0.5  # Should have some entropy
        assert ba.unique_values > 1

        # Analyze byte 4 (counter)
        ba = MessageAnalyzer.analyze_byte(filtered, 4)
        assert not ba.is_constant
        assert ba.unique_values > 1

    def test_suggest_signal_boundaries(self, sample_can_messages):
        """Test signal boundary suggestions."""

        filtered = sample_can_messages.filter_by_id(0x280)

        # Analyze all bytes
        byte_analyses = []
        for i in range(8):
            ba = MessageAnalyzer.analyze_byte(filtered, i)
            byte_analyses.append(ba)

        # Suggest boundaries
        suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

        assert len(suggestions) > 0
        # Should detect the variable bytes (2-3 for RPM, 4 for counter)
        # Exact suggestions depend on implementation

    def test_analyze_message_id(self, sample_can_messages):
        """Test complete message ID analysis."""
        analysis = MessageAnalyzer.analyze_message_id(sample_can_messages, 0x280)

        assert analysis.arbitration_id == 0x280
        assert analysis.message_count == 100
        assert analysis.frequency_hz == pytest.approx(100.0, rel=0.1)
        assert analysis.period_ms == pytest.approx(10.0, rel=0.1)

        # Should have 8 byte analyses
        assert len(analysis.byte_analyses) == 8

        # Check constant bytes
        assert analysis.byte_analyses[0].is_constant  # Byte 0 = 0xAA
        assert analysis.byte_analyses[1].is_constant  # Byte 1 = 0xBB

        # Check variable bytes
        assert not analysis.byte_analyses[2].is_constant  # RPM high byte
        assert not analysis.byte_analyses[4].is_constant  # Counter

        # Should detect counter in byte 4
        assert len(analysis.detected_counters) > 0

    def test_analyze_counter_message(self, counter_messages):
        """Test analyzing message with counter."""
        analysis = MessageAnalyzer.analyze_message_id(counter_messages, 0x100)

        # Should detect counter in byte 0
        assert len(analysis.detected_counters) > 0
        counter = analysis.detected_counters[0]
        assert counter.byte_position == 0
        assert counter.increment == 1
        assert counter.wraps_at == 255

    def test_analyze_checksum_message(self, checksum_messages):
        """Test analyzing message with checksum."""
        analysis = MessageAnalyzer.analyze_message_id(checksum_messages, 0x200)

        # Checksum detection happens separately via ChecksumDetector
        # This test just verifies analysis completes
        assert analysis.message_count == 50
        assert len(analysis.byte_analyses) == 8


@pytest.mark.unit
@pytest.mark.analyzer
class TestByteAnalysis:
    """Tests for byte-level statistical analysis."""

    def test_constant_byte(self, sample_can_messages):
        """Test analysis of constant byte."""
        filtered = sample_can_messages.filter_by_id(0x280)
        ba = MessageAnalyzer.analyze_byte(filtered, 0)  # Byte 0 = 0xAA

        assert ba.is_constant
        assert ba.min_value == 0xAA
        assert ba.max_value == 0xAA
        assert ba.unique_values == 1
        assert ba.change_rate == pytest.approx(0.0)

    def test_variable_byte(self, engine_rpm_messages):
        """Test analysis of variable byte."""
        ba = MessageAnalyzer.analyze_byte(engine_rpm_messages, 2)  # RPM high byte

        assert not ba.is_constant
        assert ba.unique_values > 1
        assert ba.entropy > 0.0
        assert ba.std > 0.0


@pytest.mark.unit
@pytest.mark.analyzer
class TestSignalBoundarySuggestion:
    """Tests for signal boundary suggestion."""

    def test_suggest_16bit_signal(self, engine_rpm_messages):
        """Test suggesting boundaries for 16-bit signal."""
        # Analyze bytes
        byte_analyses = []
        for i in range(8):
            ba = MessageAnalyzer.analyze_byte(engine_rpm_messages, i)
            byte_analyses.append(ba)

        suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

        # Should suggest bytes 2-3 as a signal
        assert len(suggestions) > 0

        # Find suggestion that includes bytes 2-3
        rpm_suggestion = None
        for sug in suggestions:
            if sug["start_byte"] == 2 and sug["num_bytes"] == 2:
                rpm_suggestion = sug
                break

        assert rpm_suggestion is not None
        assert rpm_suggestion["length_bits"] == 16
