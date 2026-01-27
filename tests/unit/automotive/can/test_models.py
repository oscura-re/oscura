"""Comprehensive test suite for CAN data models.

Tests cover CANMessage, CANMessageList, SignalDefinition, ByteAnalysis,
CounterPattern, ChecksumInfo, MessageAnalysis, and DecodedSignal.
"""

from __future__ import annotations

import pytest

# Module under test
try:
    from oscura.automotive.can.models import (
        ByteAnalysis,
        CANMessage,
        CANMessageList,
        ChecksumInfo,
        CounterPattern,
        DecodedSignal,
        MessageAnalysis,
        SignalDefinition,
    )

    HAS_CAN = True
except ImportError:
    HAS_CAN = False

pytestmark = pytest.mark.skipif(not HAS_CAN, reason="CAN modules not available")


# ============================================================================
# CANMessage Tests
# ============================================================================


def test_can_message_creation():
    """Test creating CAN message."""
    msg = CANMessage(
        arbitration_id=0x100,
        timestamp=1.234567,
        data=bytes([0x01, 0x02, 0x03, 0x04]),
        is_extended=False,
    )

    assert msg.arbitration_id == 0x100
    assert msg.timestamp == 1.234567
    assert msg.data == bytes([0x01, 0x02, 0x03, 0x04])
    assert msg.dlc == 4


def test_can_message_dlc():
    """Test DLC property matches data length."""
    msg = CANMessage(arbitration_id=0x200, timestamp=0.0, data=bytes([0, 1, 2, 3, 4, 5, 6, 7]))

    assert msg.dlc == 8


def test_can_message_extended_id():
    """Test extended ID message."""
    msg = CANMessage(arbitration_id=0x12345678, timestamp=0.0, data=bytes([0]), is_extended=True)

    assert msg.is_extended is True
    assert msg.arbitration_id == 0x12345678


def test_can_message_can_fd():
    """Test CAN-FD message."""
    msg = CANMessage(
        arbitration_id=0x100,
        timestamp=0.0,
        data=bytes(range(64)),
        is_fd=True,  # 64 bytes
    )

    assert msg.is_fd is True
    assert msg.dlc == 64


def test_can_message_repr():
    """Test message representation."""
    msg = CANMessage(arbitration_id=0x123, timestamp=1.5, data=bytes([0xAA, 0xBB]))

    repr_str = repr(msg)

    assert "0x123" in repr_str
    assert "1.5" in repr_str
    assert "AABB" in repr_str


def test_can_message_data_conversion():
    """Test automatic conversion of data to bytes."""
    msg = CANMessage(arbitration_id=0x100, timestamp=0.0, data=[0x01, 0x02, 0x03])  # type: ignore[arg-type]

    assert isinstance(msg.data, bytes)
    assert msg.data == bytes([0x01, 0x02, 0x03])


# ============================================================================
# CANMessageList Tests
# ============================================================================


def test_can_message_list_creation():
    """Test creating empty message list."""
    msg_list = CANMessageList()

    assert len(msg_list) == 0


def test_can_message_list_append():
    """Test appending messages."""
    msg_list = CANMessageList()

    msg1 = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0]))
    msg2 = CANMessage(arbitration_id=0x200, timestamp=0.1, data=bytes([1]))

    msg_list.append(msg1)
    msg_list.append(msg2)

    assert len(msg_list) == 2


def test_can_message_list_iteration():
    """Test iterating over messages."""
    messages = [CANMessage(arbitration_id=i, timestamp=i * 0.1, data=bytes([i])) for i in range(5)]
    msg_list = CANMessageList(messages=messages)

    count = 0
    for msg in msg_list:
        assert isinstance(msg, CANMessage)
        count += 1

    assert count == 5


def test_can_message_list_indexing():
    """Test indexing and slicing."""
    messages = [CANMessage(arbitration_id=i, timestamp=i * 0.1, data=bytes([i])) for i in range(10)]
    msg_list = CANMessageList(messages=messages)

    # Single index
    assert msg_list[0].arbitration_id == 0
    assert msg_list[5].arbitration_id == 5

    # Slicing
    subset = msg_list[2:5]
    assert len(subset) == 3


def test_can_message_list_filter_by_id():
    """Test filtering messages by ID."""
    messages = [
        CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0])),
        CANMessage(arbitration_id=0x200, timestamp=0.1, data=bytes([1])),
        CANMessage(arbitration_id=0x100, timestamp=0.2, data=bytes([2])),
        CANMessage(arbitration_id=0x300, timestamp=0.3, data=bytes([3])),
    ]
    msg_list = CANMessageList(messages=messages)

    filtered = msg_list.filter_by_id(0x100)

    assert len(filtered) == 2
    assert all(msg.arbitration_id == 0x100 for msg in filtered)


def test_can_message_list_unique_ids():
    """Test getting unique message IDs."""
    messages = [
        CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0])),
        CANMessage(arbitration_id=0x200, timestamp=0.1, data=bytes([1])),
        CANMessage(arbitration_id=0x100, timestamp=0.2, data=bytes([2])),
    ]
    msg_list = CANMessageList(messages=messages)

    unique_ids = msg_list.unique_ids()

    assert unique_ids == {0x100, 0x200}


def test_can_message_list_time_range():
    """Test getting time range."""
    messages = [
        CANMessage(arbitration_id=0x100, timestamp=1.0, data=bytes([0])),
        CANMessage(arbitration_id=0x100, timestamp=5.5, data=bytes([1])),
        CANMessage(arbitration_id=0x100, timestamp=3.2, data=bytes([2])),
    ]
    msg_list = CANMessageList(messages=messages)

    start, end = msg_list.time_range()

    assert start == 1.0
    assert end == 5.5


def test_can_message_list_time_range_empty():
    """Test time range of empty list."""
    msg_list = CANMessageList()

    start, end = msg_list.time_range()

    assert start == 0.0
    assert end == 0.0


# ============================================================================
# SignalDefinition Tests
# ============================================================================


def test_signal_definition_creation():
    """Test creating signal definition."""
    signal = SignalDefinition(
        name="EngineRPM",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=0.25,
        offset=0.0,
        unit="rpm",
        min_value=0.0,
        max_value=8000.0,
    )

    assert signal.name == "EngineRPM"
    assert signal.length == 16
    assert signal.scale == 0.25


def test_signal_definition_defaults():
    """Test signal definition with default values."""
    signal = SignalDefinition(name="TestSignal", start_bit=0, length=8)

    assert signal.byte_order == "big_endian"
    assert signal.value_type == "unsigned"
    assert signal.scale == 1.0
    assert signal.offset == 0.0


def test_signal_definition_start_byte():
    """Test start_byte property calculation."""
    signal = SignalDefinition(name="Test", start_bit=24, length=8)

    assert signal.start_byte == 3


def test_signal_definition_decode_unsigned():
    """Test decoding unsigned signal."""
    signal = SignalDefinition(
        name="Speed",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
    )

    data = bytes([0x01, 0x00, 0, 0, 0, 0, 0, 0])  # Value = 256
    value = signal.decode(data)

    assert value == 256.0


def test_signal_definition_decode_with_scale():
    """Test decoding with scale and offset."""
    signal = SignalDefinition(
        name="Temperature",
        start_bit=0,
        length=8,
        scale=0.5,
        offset=-40.0,
        unit="°C",
    )

    data = bytes([100, 0, 0, 0, 0, 0, 0, 0])  # Raw = 100
    value = signal.decode(data)

    # value = 100 * 0.5 + (-40) = 10
    assert value == pytest.approx(10.0)


# ============================================================================
# ByteAnalysis Tests
# ============================================================================


def test_byte_analysis_creation():
    """Test creating ByteAnalysis instance."""
    analysis = ByteAnalysis(
        position=0,
        entropy=4.5,
        min_value=0,
        max_value=255,
        mean=128.0,
        std=50.0,
        is_constant=False,
        unique_values=200,
        most_common_value=127,
        change_rate=0.8,
    )

    assert analysis.position == 0
    assert analysis.entropy == 4.5
    assert analysis.is_constant is False


# ============================================================================
# CounterPattern Tests
# ============================================================================


def test_counter_pattern_creation():
    """Test creating CounterPattern instance."""
    counter = CounterPattern(
        byte_position=0, increment=1, wraps_at=255, confidence=0.95, pattern_type="counter"
    )

    assert counter.byte_position == 0
    assert counter.increment == 1
    assert counter.wraps_at == 255
    assert counter.pattern_type == "counter"


# ============================================================================
# ChecksumInfo Tests
# ============================================================================


def test_checksum_info_creation():
    """Test creating ChecksumInfo instance."""
    checksum = ChecksumInfo(
        algorithm="xor",
        byte_position=7,
        covered_bytes=[0, 1, 2, 3, 4, 5, 6],
        confidence=0.98,
    )

    assert checksum.algorithm == "xor"
    assert checksum.byte_position == 7
    assert checksum.confidence == 0.98


# ============================================================================
# MessageAnalysis Tests
# ============================================================================


def test_message_analysis_creation():
    """Test creating MessageAnalysis instance."""
    byte_analyses = [
        ByteAnalysis(
            position=i,
            entropy=0.0,
            min_value=0,
            max_value=0,
            mean=0.0,
            std=0.0,
            is_constant=True,
            unique_values=1,
            most_common_value=0,
            change_rate=0.0,
        )
        for i in range(8)
    ]

    analysis = MessageAnalysis(
        arbitration_id=0x100,
        message_count=50,
        frequency_hz=100.0,
        period_ms=10.0,
        period_jitter_ms=0.5,
        byte_analyses=byte_analyses,
        detected_counters=[],
        detected_checksum=None,
        suggested_signals=[],
        correlations={},
    )

    assert analysis.arbitration_id == 0x100
    assert analysis.message_count == 50
    assert len(analysis.byte_analyses) == 8


# ============================================================================
# DecodedSignal Tests
# ============================================================================


def test_decoded_signal_creation():
    """Test creating DecodedSignal instance."""
    signal = DecodedSignal(
        name="RPM",
        timestamp=1.5,
        value=2500.0,
        unit="rpm",
        raw_value=10000,
    )

    assert signal.name == "RPM"
    assert signal.value == 2500.0
    assert signal.raw_value == 10000


# ============================================================================
# Edge Cases
# ============================================================================


def test_can_message_empty_data():
    """Test CAN message with empty data."""
    msg = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([]))

    assert msg.dlc == 0
    assert msg.data == bytes([])


def test_can_message_list_filter_nonexistent_id():
    """Test filtering for ID that doesn't exist."""
    messages = [CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0]))]
    msg_list = CANMessageList(messages=messages)

    filtered = msg_list.filter_by_id(0x999)

    assert len(filtered) == 0


def test_signal_definition_decode_insufficient_data():
    """Test decoding when data is too short."""
    signal = SignalDefinition(name="Test", start_bit=16, length=16)

    data = bytes([0x01])  # Too short

    with pytest.raises((IndexError, ValueError)):
        signal.decode(data)
