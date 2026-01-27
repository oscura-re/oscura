"""Comprehensive test suite for CAN signal correlation analysis.

Tests cover signal-to-signal correlation, byte-to-byte correlation,
message correlation discovery, and time-shifted correlation analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

# Module under test
try:
    from oscura.automotive.can.correlation import CorrelationAnalyzer
    from oscura.automotive.can.models import CANMessage, CANMessageList, SignalDefinition
    from oscura.automotive.can.session import CANSession

    HAS_CAN = True
except ImportError:
    HAS_CAN = False

pytestmark = pytest.mark.skipif(not HAS_CAN, reason="CAN modules not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def correlated_messages():
    """Create two message streams with correlated signals.

    ID 0x100: byte 0-1 contains RPM (0-5000)
    ID 0x200: byte 0-1 contains SPEED (0-250), correlated with RPM
    """
    messages = []

    for i in range(50):
        timestamp = i * 0.01

        # RPM increases from 1000 to 3000
        rpm = 1000 + (i * 40)
        rpm_bytes = rpm.to_bytes(2, byteorder="big")
        data_rpm = rpm_bytes + bytes([0, 0, 0, 0, 0, 0])
        msg_rpm = CANMessage(
            arbitration_id=0x100,
            timestamp=timestamp,
            data=data_rpm,
            is_extended=False,
        )
        messages.append(msg_rpm)

        # Speed correlated with RPM: speed = rpm / 20
        speed = rpm // 20
        speed_bytes = speed.to_bytes(2, byteorder="big")
        data_speed = speed_bytes + bytes([0, 0, 0, 0, 0, 0])
        msg_speed = CANMessage(
            arbitration_id=0x200,
            timestamp=timestamp + 0.001,  # Slight offset
            data=data_speed,
            is_extended=False,
        )
        messages.append(msg_speed)

    return CANMessageList(messages=messages)


@pytest.fixture
def uncorrelated_messages():
    """Create two message streams with uncorrelated signals."""
    messages = []
    rng = np.random.default_rng(42)

    for i in range(50):
        timestamp = i * 0.01

        # Random data for ID 0x300
        data1 = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
        msg1 = CANMessage(
            arbitration_id=0x300,
            timestamp=timestamp,
            data=data1,
            is_extended=False,
        )
        messages.append(msg1)

        # Different random data for ID 0x400
        data2 = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
        msg2 = CANMessage(
            arbitration_id=0x400,
            timestamp=timestamp + 0.001,
            data=data2,
            is_extended=False,
        )
        messages.append(msg2)

    return CANMessageList(messages=messages)


@pytest.fixture
def signal_def_rpm():
    """Signal definition for RPM in bytes 0-1."""
    return SignalDefinition(
        name="RPM",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
        unit="rpm",
    )


@pytest.fixture
def signal_def_speed():
    """Signal definition for speed in bytes 0-1."""
    return SignalDefinition(
        name="Speed",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
        unit="km/h",
    )


# ============================================================================
# Signal Correlation Tests
# ============================================================================


def test_correlate_signals_perfect_correlation(
    correlated_messages, signal_def_rpm, signal_def_speed
):
    """Test correlation of perfectly correlated signals."""
    rpm_messages = correlated_messages.filter_by_id(0x100)
    speed_messages = correlated_messages.filter_by_id(0x200)

    result = CorrelationAnalyzer.correlate_signals(
        rpm_messages, signal_def_rpm, speed_messages, signal_def_speed
    )

    assert "correlation" in result
    assert "p_value" in result
    assert "sample_count" in result

    # Should be highly correlated (both increase linearly)
    assert result["correlation"] > 0.99
    assert result["p_value"] < 0.01  # Statistically significant
    assert result["sample_count"] > 0


def test_correlate_signals_no_correlation(uncorrelated_messages, signal_def_rpm, signal_def_speed):
    """Test correlation of uncorrelated signals."""
    msg1 = uncorrelated_messages.filter_by_id(0x300)
    msg2 = uncorrelated_messages.filter_by_id(0x400)

    result = CorrelationAnalyzer.correlate_signals(msg1, signal_def_rpm, msg2, signal_def_speed)

    # Random data should have low correlation
    assert abs(result["correlation"]) < 0.5
    assert result["sample_count"] > 0


def test_correlate_signals_empty_messages():
    """Test correlation with empty message lists."""
    empty = CANMessageList(messages=[])
    signal_def = SignalDefinition(
        name="Test", start_bit=0, length=16, byte_order="big_endian", value_type="unsigned"
    )

    result = CorrelationAnalyzer.correlate_signals(empty, signal_def, empty, signal_def)

    assert result["correlation"] == 0.0
    assert result["p_value"] == 1.0
    assert result["sample_count"] == 0


def test_correlate_signals_insufficient_samples():
    """Test correlation with insufficient valid samples."""
    # Only 1 message each
    msg1 = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0, 0, 0, 0]))
    msg2 = CANMessage(arbitration_id=0x200, timestamp=0.0, data=bytes([0, 0, 0, 0]))

    messages1 = CANMessageList(messages=[msg1])
    messages2 = CANMessageList(messages=[msg2])

    signal_def = SignalDefinition(
        name="Test", start_bit=0, length=16, byte_order="big_endian", value_type="unsigned"
    )

    result = CorrelationAnalyzer.correlate_signals(messages1, signal_def, messages2, signal_def)

    # Need at least 2 samples for correlation
    assert result["correlation"] == 0.0
    assert result["sample_count"] < 2


def test_correlate_signals_time_alignment(correlated_messages, signal_def_rpm, signal_def_speed):
    """Test that signals are properly time-aligned."""
    rpm_messages = correlated_messages.filter_by_id(0x100)
    speed_messages = correlated_messages.filter_by_id(0x200)

    # Messages have 1ms offset - should still align
    result = CorrelationAnalyzer.correlate_signals(
        rpm_messages, signal_def_rpm, speed_messages, signal_def_speed, max_time_shift=0.01
    )

    assert result["sample_count"] > 40  # Should align most messages
    assert result["correlation"] > 0.99


def test_correlate_signals_negative_correlation():
    """Test detection of negative correlation."""
    messages = []

    for i in range(30):
        timestamp = i * 0.01

        # Signal 1 increases
        value1 = i * 10
        data1 = value1.to_bytes(2, byteorder="big") + bytes([0, 0, 0, 0, 0, 0])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages.append(msg1)

        # Signal 2 decreases (perfect negative correlation)
        value2 = (30 - i) * 10
        data2 = value2.to_bytes(2, byteorder="big") + bytes([0, 0, 0, 0, 0, 0])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages.append(msg2)

    msg_list = CANMessageList(messages=messages)
    signal_def = SignalDefinition(
        name="Test", start_bit=0, length=16, byte_order="big_endian", value_type="unsigned"
    )

    result = CorrelationAnalyzer.correlate_signals(
        msg_list.filter_by_id(0x100),
        signal_def,
        msg_list.filter_by_id(0x200),
        signal_def,
    )

    # Should have strong negative correlation
    assert result["correlation"] < -0.99


# ============================================================================
# Byte Correlation Tests
# ============================================================================


def test_correlate_bytes_identical_sequences():
    """Test byte correlation with identical sequences."""
    messages1 = []
    messages2 = []

    for i in range(30):
        data = bytes([i & 0xFF, 0, 0, 0, 0, 0, 0, 0])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data)
        msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data)
        messages1.append(msg1)
        messages2.append(msg2)

    list1 = CANMessageList(messages=messages1)
    list2 = CANMessageList(messages=messages2)

    correlation = CorrelationAnalyzer.correlate_bytes(list1, 0, list2, 0)

    # Perfect correlation
    assert correlation == pytest.approx(1.0, abs=0.01)


def test_correlate_bytes_no_correlation():
    """Test byte correlation with random data."""
    rng = np.random.default_rng(42)
    messages1 = []
    messages2 = []

    for i in range(30):
        data1 = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
        data2 = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
        msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
        msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)
        messages1.append(msg1)
        messages2.append(msg2)

    list1 = CANMessageList(messages=messages1)
    list2 = CANMessageList(messages=messages2)

    correlation = CorrelationAnalyzer.correlate_bytes(list1, 0, list2, 0)

    # Low correlation expected
    assert abs(correlation) < 0.5


def test_correlate_bytes_insufficient_data():
    """Test byte correlation with insufficient data."""
    msg1 = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0x42]))
    list1 = CANMessageList(messages=[msg1])

    correlation = CorrelationAnalyzer.correlate_bytes(list1, 0, list1, 0)

    # Single sample - cannot compute correlation
    assert correlation == 0.0


def test_correlate_bytes_zero_variance():
    """Test byte correlation with constant values (zero variance)."""
    messages1 = []
    messages2 = []

    for i in range(30):
        # Constant value in byte 0
        data1 = bytes([0x42, 0, 0, 0, 0, 0, 0, 0])
        data2 = bytes([0x99, 0, 0, 0, 0, 0, 0, 0])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
        msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)
        messages1.append(msg1)
        messages2.append(msg2)

    list1 = CANMessageList(messages=messages1)
    list2 = CANMessageList(messages=messages2)

    correlation = CorrelationAnalyzer.correlate_bytes(list1, 0, list2, 0)

    # Zero variance - cannot compute correlation
    assert correlation == 0.0


def test_correlate_bytes_different_lengths():
    """Test byte correlation with different message list lengths."""
    messages1 = [
        CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=bytes([i & 0xFF, 0, 0, 0]))
        for i in range(50)
    ]
    messages2 = [
        CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=bytes([i & 0xFF, 0, 0, 0]))
        for i in range(30)
    ]

    list1 = CANMessageList(messages=messages1)
    list2 = CANMessageList(messages=messages2)

    correlation = CorrelationAnalyzer.correlate_bytes(list1, 0, list2, 0)

    # Should truncate to shorter length and compute correlation
    assert correlation == pytest.approx(1.0, abs=0.01)


def test_correlate_bytes_out_of_range():
    """Test byte correlation with byte position beyond message length."""
    messages = [
        CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=bytes([0x42, 0x43]))
        for i in range(10)
    ]
    msg_list = CANMessageList(messages=messages)

    correlation = CorrelationAnalyzer.correlate_bytes(msg_list, 10, msg_list, 10)

    # No data at position 10
    assert correlation == 0.0


# ============================================================================
# Message Correlation Discovery Tests
# ============================================================================


def test_find_correlated_messages_basic(correlated_messages):
    """Test finding correlated messages in a session."""
    session = CANSession()
    session._messages = correlated_messages

    # Find messages correlated with 0x100 (RPM)
    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.7
    )

    # Should find 0x200 (speed) as correlated
    assert 0x200 in correlations
    assert abs(correlations[0x200]) > 0.7


def test_find_correlated_messages_no_correlations(uncorrelated_messages):
    """Test finding correlated messages when none exist."""
    session = CANSession()
    session._messages = uncorrelated_messages

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x300, threshold=0.7
    )

    # Should find no strong correlations
    assert len(correlations) == 0


def test_find_correlated_messages_empty_session():
    """Test finding correlations in empty session."""
    session = CANSession()

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.7
    )

    assert len(correlations) == 0


def test_find_correlated_messages_nonexistent_id():
    """Test finding correlations for non-existent message ID."""
    session = CANSession()
    msg = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0, 0, 0, 0]))
    session._messages = CANMessageList(messages=[msg])

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x999, threshold=0.7
    )

    assert len(correlations) == 0


def test_find_correlated_messages_threshold_filtering():
    """Test that threshold parameter filters results correctly."""
    messages = []

    # Create weak correlation
    for i in range(30):
        timestamp = i * 0.01

        # Signal 1: increases linearly
        data1 = bytes([i & 0xFF, 0, 0, 0, 0, 0, 0, 0])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages.append(msg1)

        # Signal 2: increases but with noise (weaker correlation)
        value2 = (i + (i % 5)) & 0xFF
        data2 = bytes([value2, 0, 0, 0, 0, 0, 0, 0])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages.append(msg2)

    session = CANSession()
    session._messages = CANMessageList(messages=messages)

    # Lower threshold - should find correlation
    correlations_low = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.5
    )

    # Higher threshold - might not find it
    correlations_high = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.95
    )

    # Lower threshold should find more correlations
    assert len(correlations_low) >= len(correlations_high)


def test_find_correlated_messages_excludes_self():
    """Test that message doesn't correlate with itself."""
    messages = [
        CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=bytes([i & 0xFF, 0, 0, 0]))
        for i in range(30)
    ]

    session = CANSession()
    session._messages = CANMessageList(messages=messages)

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.0
    )

    # Should not include itself
    assert 0x100 not in correlations


def test_find_correlated_messages_multiple_byte_positions():
    """Test correlation discovery across multiple byte positions."""
    messages = []

    for i in range(30):
        timestamp = i * 0.01

        # Message 1: value in byte 0
        data1 = bytes([i & 0xFF, 0, 0, 0, 0, 0, 0, 0])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages.append(msg1)

        # Message 2: same value but in byte 3
        data2 = bytes([0, 0, 0, i & 0xFF, 0, 0, 0, 0])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages.append(msg2)

    session = CANSession()
    session._messages = CANMessageList(messages=messages)

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.7
    )

    # Should find correlation even though bytes are at different positions
    assert 0x200 in correlations
    assert abs(correlations[0x200]) > 0.9


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_correlate_signals_decode_errors():
    """Test correlation handles signal decode errors gracefully."""
    messages = []
    for i in range(10):
        # Data too short for 16-bit signal
        data = bytes([0x42])
        msg = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data)
        messages.append(msg)

    msg_list = CANMessageList(messages=messages)
    signal_def = SignalDefinition(
        name="Test",
        start_bit=8,  # Beyond available data
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
    )

    result = CorrelationAnalyzer.correlate_signals(msg_list, signal_def, msg_list, signal_def)

    # Should handle decode errors and return zero correlation
    assert result["correlation"] == 0.0
    assert result["sample_count"] == 0


def test_correlate_bytes_empty_lists():
    """Test byte correlation with empty message lists."""
    empty = CANMessageList(messages=[])

    correlation = CorrelationAnalyzer.correlate_bytes(empty, 0, empty, 0)

    assert correlation == 0.0


def test_find_correlated_messages_single_message():
    """Test correlation discovery with single message per ID."""
    msg1 = CANMessage(arbitration_id=0x100, timestamp=0.0, data=bytes([0x42, 0, 0, 0]))
    msg2 = CANMessage(arbitration_id=0x200, timestamp=0.0, data=bytes([0x42, 0, 0, 0]))

    session = CANSession()
    session._messages = CANMessageList(messages=[msg1, msg2])

    correlations = CorrelationAnalyzer.find_correlated_messages(
        session, arbitration_id=0x100, threshold=0.5
    )

    # Cannot compute correlation with single sample
    assert len(correlations) == 0
