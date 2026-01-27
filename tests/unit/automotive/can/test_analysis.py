"""Comprehensive test suite for CAN message analysis algorithms.

Tests cover entropy calculation, counter detection, byte analysis, signal boundary
suggestion, and complete message analysis workflows.
"""

from __future__ import annotations

import numpy as np
import pytest

# Module under test
try:
    from oscura.automotive.can.analysis import MessageAnalyzer
    from oscura.automotive.can.models import CANMessage, CANMessageList

    HAS_CAN = True
except ImportError:
    HAS_CAN = False

pytestmark = pytest.mark.skipif(not HAS_CAN, reason="CAN modules not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def constant_messages():
    """Messages with constant data (low entropy)."""
    messages = []
    for i in range(20):
        data = bytes([0x12, 0x34, 0x56, 0x78, 0xAA, 0xBB, 0xCC, 0xDD])
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)
    return CANMessageList(messages=messages)


@pytest.fixture
def counter_messages():
    """Messages with counter in byte 0."""
    messages = []
    for i in range(50):
        counter = i % 256
        data = bytes([counter, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)
    return CANMessageList(messages=messages)


@pytest.fixture
def varying_messages():
    """Messages with high entropy data."""
    messages = []
    rng = np.random.default_rng(42)
    for i in range(20):
        data = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)
    return CANMessageList(messages=messages)


@pytest.fixture
def mixed_signal_messages():
    """Messages with mixed constant and varying bytes."""
    messages = []
    for i in range(30):
        # Bytes 0-1: constant
        # Bytes 2-3: counter
        # Bytes 4-7: varying
        counter = i % 256
        varying_byte = (i * 17) % 256
        data = bytes([0xAA, 0xBB, counter, 0x00, varying_byte, varying_byte, 0x00, 0x00])
        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)
    return CANMessageList(messages=messages)


# ============================================================================
# Entropy Calculation Tests
# ============================================================================


def test_calculate_entropy_constant_value():
    """Test entropy of constant values is 0."""
    values = [0x42] * 100
    entropy = MessageAnalyzer.calculate_entropy(values)
    assert entropy == pytest.approx(0.0, abs=1e-6)


def test_calculate_entropy_two_values():
    """Test entropy of two equally likely values."""
    values = [0x00, 0xFF] * 50
    entropy = MessageAnalyzer.calculate_entropy(values)
    # Entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
    assert entropy == pytest.approx(1.0, abs=0.01)


def test_calculate_entropy_uniform_distribution():
    """Test entropy of uniform distribution."""
    # All 256 values equally likely
    values = list(range(256))
    entropy = MessageAnalyzer.calculate_entropy(values)
    # Entropy = log2(256) = 8.0
    assert entropy == pytest.approx(8.0, abs=0.01)


def test_calculate_entropy_empty_list():
    """Test entropy of empty list is 0."""
    entropy = MessageAnalyzer.calculate_entropy([])
    assert entropy == 0.0


def test_calculate_entropy_single_value():
    """Test entropy of single value."""
    entropy = MessageAnalyzer.calculate_entropy([0x42])
    assert entropy == 0.0


# ============================================================================
# Counter Detection Tests
# ============================================================================


def test_detect_counter_simple_increment():
    """Test detection of simple counter (increment by 1)."""
    values = list(range(50))
    counter = MessageAnalyzer.detect_counter(values)

    assert counter is not None
    assert counter.increment == 1
    assert counter.pattern_type == "counter"
    assert counter.confidence > 0.9


def test_detect_counter_with_wraparound():
    """Test counter detection with wraparound at 255."""
    values = list(range(250, 256)) + list(range(10))
    counter = MessageAnalyzer.detect_counter(values)

    assert counter is not None
    assert counter.increment == 1
    assert counter.wraps_at == 255


def test_detect_counter_increment_by_2():
    """Test detection of counter incrementing by 2."""
    values = list(range(0, 100, 2))
    counter = MessageAnalyzer.detect_counter(values)

    assert counter is not None
    assert counter.increment == 2
    assert counter.pattern_type == "counter"


def test_detect_counter_increment_by_4():
    """Test detection of counter incrementing by 4."""
    values = list(range(0, 100, 4))
    counter = MessageAnalyzer.detect_counter(values)

    assert counter is not None
    assert counter.increment == 4


def test_detect_counter_too_few_samples():
    """Test that counter detection fails with too few samples."""
    values = [0, 1]
    counter = MessageAnalyzer.detect_counter(values)
    assert counter is None


def test_detect_counter_random_values():
    """Test that random values don't get detected as counter."""
    rng = np.random.default_rng(42)
    values = list(rng.integers(0, 256, size=50))
    counter = MessageAnalyzer.detect_counter(values)
    # Should either be None or very low confidence
    if counter is not None:
        assert counter.confidence < 0.7


def test_detect_counter_wraps_at_15():
    """Test counter wrapping at 15 (4-bit counter)."""
    values = list(range(16)) * 3  # Repeat pattern
    counter = MessageAnalyzer.detect_counter(values, max_value=15)

    assert counter is not None
    assert counter.increment == 1
    assert counter.wraps_at == 15


def test_detect_counter_toggle_pattern():
    """Test detection of toggle pattern (not simple counter)."""
    values = [0, 1] * 50  # Alternating
    counter = MessageAnalyzer.detect_counter(values)

    # Should detect as sequence with high confidence
    if counter is not None:
        assert counter.pattern_type in ["counter", "sequence"]


# ============================================================================
# Byte Analysis Tests
# ============================================================================


def test_analyze_byte_constant_value(constant_messages):
    """Test analysis of constant byte position."""
    analysis = MessageAnalyzer.analyze_byte(constant_messages, byte_position=0)

    assert analysis.position == 0
    assert analysis.is_constant is True
    assert analysis.unique_values == 1
    assert analysis.min_value == 0x12
    assert analysis.max_value == 0x12
    assert analysis.entropy == pytest.approx(0.0, abs=1e-6)
    assert analysis.change_rate == 0.0


def test_analyze_byte_counter(counter_messages):
    """Test analysis of counter byte."""
    analysis = MessageAnalyzer.analyze_byte(counter_messages, byte_position=0)

    assert analysis.position == 0
    assert analysis.is_constant is False
    assert analysis.unique_values == 50
    assert analysis.min_value == 0
    assert analysis.max_value == 49
    assert analysis.entropy > 0.0
    assert analysis.change_rate > 0.9  # Changes almost every message


def test_analyze_byte_varying_data(varying_messages):
    """Test analysis of varying byte position."""
    analysis = MessageAnalyzer.analyze_byte(varying_messages, byte_position=0)

    assert analysis.position == 0
    assert analysis.is_constant is False
    assert analysis.unique_values > 1
    assert analysis.entropy > 0.0


def test_analyze_byte_out_of_range(counter_messages):
    """Test analysis of byte position beyond message length."""
    analysis = MessageAnalyzer.analyze_byte(counter_messages, byte_position=10)

    # Should return safe defaults
    assert analysis.position == 10
    assert analysis.is_constant is True
    assert analysis.unique_values == 0
    assert analysis.entropy == 0.0


def test_analyze_byte_statistics_accuracy(counter_messages):
    """Test statistical calculations are accurate."""
    analysis = MessageAnalyzer.analyze_byte(counter_messages, byte_position=0)

    # Counter from 0-49
    assert analysis.mean == pytest.approx(24.5, abs=0.1)  # Mean of 0..49
    assert analysis.std > 0.0  # Should have non-zero standard deviation


# ============================================================================
# Signal Boundary Suggestion Tests
# ============================================================================


def test_suggest_signal_boundaries_all_constant():
    """Test signal boundary suggestion with all constant bytes."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=i,
            entropy=0.0,
            min_value=0xAA,
            max_value=0xAA,
            mean=0xAA,
            std=0.0,
            is_constant=True,
            unique_values=1,
            most_common_value=0xAA,
            change_rate=0.0,
        )
        for i in range(8)
    ]

    suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

    # Should suggest no signals (all constant)
    assert len(suggestions) == 0


def test_suggest_signal_boundaries_single_varying_byte():
    """Test signal boundary suggestion with single varying byte."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = []
    for i in range(8):
        is_varying = i == 3  # Only byte 3 varies
        byte_analyses.append(
            ByteAnalysis(
                position=i,
                entropy=4.0 if is_varying else 0.0,
                min_value=0 if is_varying else 0xAA,
                max_value=255 if is_varying else 0xAA,
                mean=128.0 if is_varying else 0xAA,
                std=50.0 if is_varying else 0.0,
                is_constant=not is_varying,
                unique_values=256 if is_varying else 1,
                most_common_value=0,
                change_rate=0.9 if is_varying else 0.0,
            )
        )

    suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

    assert len(suggestions) == 1
    assert suggestions[0]["start_byte"] == 3
    assert suggestions[0]["num_bytes"] == 1
    assert suggestions[0]["length_bits"] == 8


def test_suggest_signal_boundaries_contiguous_varying_bytes():
    """Test signal boundary suggestion with contiguous varying bytes."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = []
    for i in range(8):
        # Bytes 2-5 vary, others constant
        is_varying = 2 <= i <= 5
        byte_analyses.append(
            ByteAnalysis(
                position=i,
                entropy=5.0 if is_varying else 0.0,
                min_value=0 if is_varying else 0xAA,
                max_value=255 if is_varying else 0xAA,
                mean=128.0 if is_varying else 0xAA,
                std=50.0 if is_varying else 0.0,
                is_constant=not is_varying,
                unique_values=200 if is_varying else 1,
                most_common_value=0,
                change_rate=0.8 if is_varying else 0.0,
            )
        )

    suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

    assert len(suggestions) == 1
    assert suggestions[0]["start_byte"] == 2
    assert suggestions[0]["num_bytes"] == 4
    assert suggestions[0]["length_bits"] == 32


def test_suggest_signal_boundaries_multiple_groups():
    """Test signal boundary suggestion with multiple separated groups."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = []
    for i in range(8):
        # Bytes 0-1 vary, 2-3 constant, 4-5 vary, 6-7 constant
        is_varying = i in [0, 1, 4, 5]
        byte_analyses.append(
            ByteAnalysis(
                position=i,
                entropy=4.0 if is_varying else 0.0,
                min_value=0 if is_varying else 0xAA,
                max_value=255 if is_varying else 0xAA,
                mean=128.0 if is_varying else 0xAA,
                std=50.0 if is_varying else 0.0,
                is_constant=not is_varying,
                unique_values=150 if is_varying else 1,
                most_common_value=0,
                change_rate=0.7 if is_varying else 0.0,
            )
        )

    suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

    assert len(suggestions) == 2
    # First group: bytes 0-1
    assert suggestions[0]["start_byte"] == 0
    assert suggestions[0]["num_bytes"] == 2
    # Second group: bytes 4-5
    assert suggestions[1]["start_byte"] == 4
    assert suggestions[1]["num_bytes"] == 2


def test_suggest_signal_boundaries_type_suggestions():
    """Test that type suggestions are included."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=0,
            entropy=5.0,
            min_value=0,
            max_value=255,
            mean=128.0,
            std=50.0,
            is_constant=False,
            unique_values=200,
            most_common_value=0,
            change_rate=0.8,
        ),
        ByteAnalysis(
            position=1,
            entropy=5.0,
            min_value=0,
            max_value=255,
            mean=128.0,
            std=50.0,
            is_constant=False,
            unique_values=200,
            most_common_value=0,
            change_rate=0.8,
        ),
    ]

    suggestions = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

    assert len(suggestions) == 1
    assert "suggested_types" in suggestions[0]
    # Should suggest uint16/int16 for 2-byte signal
    assert "uint16" in suggestions[0]["suggested_types"]


# ============================================================================
# Complete Message Analysis Tests
# ============================================================================


def test_analyze_message_id_basic(counter_messages):
    """Test complete message analysis."""
    analysis = MessageAnalyzer.analyze_message_id(counter_messages, arbitration_id=0x200)

    assert analysis.arbitration_id == 0x200
    assert analysis.message_count == 50
    assert analysis.frequency_hz > 0
    assert analysis.period_ms > 0
    assert len(analysis.byte_analyses) == 8
    assert len(analysis.detected_counters) > 0


def test_analyze_message_id_timing_statistics(counter_messages):
    """Test timing statistics calculation."""
    analysis = MessageAnalyzer.analyze_message_id(counter_messages, arbitration_id=0x200)

    # Messages at 10ms intervals
    assert analysis.period_ms == pytest.approx(10.0, abs=0.1)
    assert analysis.frequency_hz == pytest.approx(100.0, abs=1.0)
    assert analysis.period_jitter_ms < 1.0  # Should be very low jitter


def test_analyze_message_id_detects_counters(counter_messages):
    """Test that counter detection works in full analysis."""
    analysis = MessageAnalyzer.analyze_message_id(counter_messages, arbitration_id=0x200)

    assert len(analysis.detected_counters) >= 1
    # Counter should be at byte position 0
    counter = analysis.detected_counters[0]
    assert counter.byte_position == 0
    assert counter.increment == 1


def test_analyze_message_id_mixed_signals(mixed_signal_messages):
    """Test analysis of messages with mixed signal types."""
    analysis = MessageAnalyzer.analyze_message_id(mixed_signal_messages, arbitration_id=0x400)

    assert analysis.message_count == 30

    # Byte 0-1 should be constant
    assert analysis.byte_analyses[0].is_constant is True
    assert analysis.byte_analyses[1].is_constant is True

    # Byte 2 should be counter
    assert analysis.byte_analyses[2].is_constant is False
    assert len(analysis.detected_counters) >= 1


def test_analyze_message_id_no_messages():
    """Test analysis when no messages exist for ID."""
    messages = CANMessageList(messages=[])

    with pytest.raises(ValueError, match="No messages found"):
        MessageAnalyzer.analyze_message_id(messages, arbitration_id=0x999)


def test_analyze_message_id_single_message():
    """Test analysis with single message."""
    msg = CANMessage(
        arbitration_id=0x123,
        timestamp=1.0,
        data=bytes([0x01, 0x02, 0x03, 0x04]),
        is_extended=False,
    )
    messages = CANMessageList(messages=[msg])

    analysis = MessageAnalyzer.analyze_message_id(messages, arbitration_id=0x123)

    assert analysis.message_count == 1
    assert len(analysis.byte_analyses) == 4
    # All bytes should be constant (only 1 sample)
    assert all(ba.is_constant for ba in analysis.byte_analyses)


# ============================================================================
# Type Suggestion Tests
# ============================================================================


def test_suggest_types_single_byte():
    """Test type suggestions for 1-byte signal."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=0,
            entropy=5.0,
            min_value=0,
            max_value=255,
            mean=128.0,
            std=50.0,
            is_constant=False,
            unique_values=200,
            most_common_value=0,
            change_rate=0.8,
        )
    ]

    types = MessageAnalyzer._suggest_types(byte_analyses)

    assert "uint8" in types
    assert "int8" in types


def test_suggest_types_two_bytes():
    """Test type suggestions for 2-byte signal."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=i,
            entropy=5.0,
            min_value=0,
            max_value=255,
            mean=128.0,
            std=50.0,
            is_constant=False,
            unique_values=200,
            most_common_value=0,
            change_rate=0.8,
        )
        for i in range(2)
    ]

    types = MessageAnalyzer._suggest_types(byte_analyses)

    assert "uint16" in types
    assert "int16" in types


def test_suggest_types_four_bytes():
    """Test type suggestions for 4-byte signal."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=i,
            entropy=5.0,
            min_value=0,
            max_value=255,
            mean=128.0,
            std=50.0,
            is_constant=False,
            unique_values=200,
            most_common_value=0,
            change_rate=0.8,
        )
        for i in range(4)
    ]

    types = MessageAnalyzer._suggest_types(byte_analyses)

    assert "uint32" in types
    assert "int32" in types
    assert "float32" in types


def test_suggest_types_automotive_percentage():
    """Test automotive-specific type suggestions (percentage)."""
    from oscura.automotive.can.models import ByteAnalysis

    byte_analyses = [
        ByteAnalysis(
            position=i,
            entropy=5.0,
            min_value=0,
            max_value=50,  # Max value suggests percentage
            mean=25.0,
            std=10.0,
            is_constant=False,
            unique_values=50,
            most_common_value=0,
            change_rate=0.7,
        )
        for i in range(2)
    ]

    types = MessageAnalyzer._suggest_types(byte_analyses)

    assert "percentage" in types


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_analyze_byte_empty_messages():
    """Test byte analysis with empty message list."""
    messages = CANMessageList(messages=[])
    analysis = MessageAnalyzer.analyze_byte(messages, byte_position=0)

    # Should return safe defaults
    assert analysis.is_constant is True
    assert analysis.entropy == 0.0


def test_calculate_entropy_all_zeros():
    """Test entropy calculation with all zeros."""
    values = [0] * 100
    entropy = MessageAnalyzer.calculate_entropy(values)
    assert entropy == 0.0


def test_detect_counter_max_value_parameter():
    """Test counter detection with custom max_value."""
    values = list(range(16)) + list(range(16))  # Wraps at 15
    counter = MessageAnalyzer.detect_counter(values, max_value=15)

    assert counter is not None
    assert counter.wraps_at == 15
