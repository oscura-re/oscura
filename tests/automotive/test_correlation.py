"""Tests for CAN signal and message correlation analysis.

This module tests correlation analysis for discovering relationships between
signals and messages in CAN bus traffic.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import numpy as np
import pytest

from oscura.automotive.can.correlation import CorrelationAnalyzer
from oscura.automotive.can.models import CANMessage, CANMessageList, SignalDefinition


@pytest.fixture
def correlated_signals() -> tuple[CANMessageList, CANMessageList]:
    """Create two message lists with strongly correlated signals.

    Returns:
        Tuple of (messages1, messages2) with correlated data.
    """
    messages1 = CANMessageList()
    messages2 = CANMessageList()

    # Create 100 messages with linearly correlated values
    # Signal 1: Value increases from 0 to 99
    # Signal 2: Value = Signal1 * 2 + 10 (perfect linear correlation)
    for i in range(100):
        timestamp = i * 0.01

        # Message 1: uint8 at byte 0
        value1 = i
        data1 = bytes([value1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages1.append(msg1)

        # Message 2: uint8 at byte 0 (correlated value)
        value2 = min(255, i * 2 + 10)
        data2 = bytes([value2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages2.append(msg2)

    return messages1, messages2


@pytest.fixture
def uncorrelated_signals() -> tuple[CANMessageList, CANMessageList]:
    """Create two message lists with uncorrelated signals.

    Returns:
        Tuple of (messages1, messages2) with uncorrelated data.
    """
    messages1 = CANMessageList()
    messages2 = CANMessageList()

    rng = np.random.default_rng(42)

    for i in range(100):
        timestamp = i * 0.01

        # Random values (no correlation)
        value1 = int(rng.integers(0, 256))
        data1 = bytes([value1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages1.append(msg1)

        value2 = int(rng.integers(0, 256))
        data2 = bytes([value2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages2.append(msg2)

    return messages1, messages2


@pytest.fixture
def inversely_correlated_signals() -> tuple[CANMessageList, CANMessageList]:
    """Create two message lists with inversely correlated signals.

    Returns:
        Tuple of (messages1, messages2) with inverse correlation.
    """
    messages1 = CANMessageList()
    messages2 = CANMessageList()

    for i in range(100):
        timestamp = i * 0.01

        # Signal 1 increases
        value1 = i
        data1 = bytes([value1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
        messages1.append(msg1)

        # Signal 2 decreases (inverse correlation)
        value2 = 99 - i
        data2 = bytes([value2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
        messages2.append(msg2)

    return messages1, messages2


class TestCorrelateSignals:
    """Tests for signal-to-signal correlation analysis."""

    def test_correlate_perfectly_correlated_signals(self, correlated_signals):
        """Test correlation of perfectly correlated signals."""
        messages1, messages2 = correlated_signals

        # Define signal definitions (uint8 at byte 0)
        signal_def1 = SignalDefinition(
            name="Signal1",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )
        signal_def2 = SignalDefinition(
            name="Signal2",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )

        result = CorrelationAnalyzer.correlate_signals(
            messages1, signal_def1, messages2, signal_def2
        )

        # Should have strong positive correlation
        assert result["correlation"] > 0.95
        assert result["p_value"] < 0.05
        assert result["sample_count"] == 100

    def test_correlate_uncorrelated_signals(self, uncorrelated_signals):
        """Test correlation of uncorrelated signals."""
        messages1, messages2 = uncorrelated_signals

        signal_def1 = SignalDefinition(
            name="Signal1",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )
        signal_def2 = SignalDefinition(
            name="Signal2",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )

        result = CorrelationAnalyzer.correlate_signals(
            messages1, signal_def1, messages2, signal_def2
        )

        # Should have low correlation
        assert abs(result["correlation"]) < 0.5

    def test_correlate_inversely_correlated_signals(self, inversely_correlated_signals):
        """Test correlation of inversely correlated signals."""
        messages1, messages2 = inversely_correlated_signals

        signal_def1 = SignalDefinition(
            name="Signal1",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )
        signal_def2 = SignalDefinition(
            name="Signal2",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )

        result = CorrelationAnalyzer.correlate_signals(
            messages1, signal_def1, messages2, signal_def2
        )

        # Should have strong negative correlation
        assert result["correlation"] < -0.95
        assert result["p_value"] < 0.05

    def test_correlate_with_insufficient_samples(self):
        """Test correlation with too few samples returns zero correlation."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        # Only 1 message each
        msg1 = CANMessage(arbitration_id=0x100, timestamp=1.0, data=bytes(8))
        messages1.append(msg1)
        msg2 = CANMessage(arbitration_id=0x200, timestamp=1.0, data=bytes(8))
        messages2.append(msg2)

        signal_def1 = SignalDefinition(
            name="Signal1",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )
        signal_def2 = SignalDefinition(
            name="Signal2",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
        )

        result = CorrelationAnalyzer.correlate_signals(
            messages1, signal_def1, messages2, signal_def2
        )

        assert result["correlation"] == 0.0
        assert result["sample_count"] == 0

    def test_correlate_with_scale_and_offset(self):
        """Test correlation with signals using scale and offset."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        for i in range(50):
            timestamp = i * 0.01

            # Signal 1: raw value i
            data1 = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg1 = CANMessage(arbitration_id=0x100, timestamp=timestamp, data=data1)
            messages1.append(msg1)

            # Signal 2: raw value i (same raw values)
            data2 = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg2 = CANMessage(arbitration_id=0x200, timestamp=timestamp, data=data2)
            messages2.append(msg2)

        # Signal 1: value = raw * 0.5
        signal_def1 = SignalDefinition(
            name="Signal1",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=0.5,
            offset=0.0,
        )

        # Signal 2: value = raw * 2.0
        signal_def2 = SignalDefinition(
            name="Signal2",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=2.0,
            offset=0.0,
        )

        result = CorrelationAnalyzer.correlate_signals(
            messages1, signal_def1, messages2, signal_def2
        )

        # Should still be perfectly correlated (linear relationship)
        assert result["correlation"] > 0.99


class TestCorrelateBytes:
    """Tests for byte-to-byte correlation analysis."""

    def test_correlate_identical_bytes(self):
        """Test correlation of identical byte positions."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        for i in range(100):
            # Same value in both messages
            value = i % 256
            data1 = bytes([value, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            data2 = bytes([value, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        assert correlation > 0.99  # Nearly perfect correlation

    def test_correlate_uncorrelated_bytes(self):
        """Test correlation of random byte positions."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        rng = np.random.default_rng(42)

        for i in range(100):
            value1 = int(rng.integers(0, 256))
            value2 = int(rng.integers(0, 256))

            data1 = bytes([value1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            data2 = bytes([value2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        # Should have low correlation
        assert abs(correlation) < 0.5

    def test_correlate_inversely_correlated_bytes(self):
        """Test correlation of inversely correlated bytes."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        for i in range(100):
            value1 = i
            value2 = 99 - i  # Inverse

            data1 = bytes([value1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            data2 = bytes([value2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        assert correlation < -0.95  # Strong negative correlation

    def test_correlate_with_constant_byte(self):
        """Test correlation when one byte is constant (zero variance)."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        for i in range(100):
            # Varying value
            data1 = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            # Constant value
            data2 = bytes([0x55, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        # Zero variance should return 0 correlation
        assert correlation == 0.0

    def test_correlate_different_byte_positions(self):
        """Test correlation between different byte positions."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        for i in range(100):
            value = i % 256

            # Same value in different positions
            data1 = bytes([value, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            data2 = bytes([0x00, 0x00, value, 0x00, 0x00, 0x00, 0x00, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        # Byte 0 of messages1 vs byte 2 of messages2
        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 2)

        assert correlation > 0.99

    def test_correlate_with_insufficient_data(self):
        """Test correlation with messages too short for byte position."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        # Messages with only 2 bytes
        for i in range(10):
            data1 = bytes([i, 0x00])
            data2 = bytes([i, 0x00])

            msg1 = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data1)
            msg2 = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data2)

            messages1.append(msg1)
            messages2.append(msg2)

        # Try to access byte 5 (doesn't exist)
        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 5, messages2, 5)

        assert correlation == 0.0

    def test_correlate_empty_message_lists(self):
        """Test correlation with empty message lists."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        assert correlation == 0.0


class TestFindCorrelatedMessages:
    """Tests for finding correlated messages in a session."""

    def test_find_correlated_messages_basic(self, sample_can_messages):
        """Test finding correlated messages in a session."""
        from oscura.automotive.can.session import CANSession

        session = CANSession(sample_can_messages)

        # Find messages correlated with 0x280 (engine RPM)
        # Should find 0x400 which also has varying signals
        correlations = CorrelationAnalyzer.find_correlated_messages(session, 0x280, threshold=0.5)

        # Check that we get a dict
        assert isinstance(correlations, dict)

    def test_find_correlated_messages_with_high_threshold(self, sample_can_messages):
        """Test that high threshold filters out weak correlations."""
        from oscura.automotive.can.session import CANSession

        session = CANSession(sample_can_messages)

        # Very high threshold should return few/no correlations
        correlations = CorrelationAnalyzer.find_correlated_messages(session, 0x280, threshold=0.99)

        # Should have no or very few correlations
        assert len(correlations) <= 1

    def test_find_correlated_with_nonexistent_id(self, sample_can_messages):
        """Test finding correlations for non-existent message ID."""
        from oscura.automotive.can.session import CANSession

        session = CANSession(sample_can_messages)

        correlations = CorrelationAnalyzer.find_correlated_messages(session, 0x999, threshold=0.7)

        # Should return empty dict
        assert correlations == {}

    def test_find_correlated_excludes_self(self, sample_can_messages):
        """Test that correlation search excludes the target ID itself."""
        from oscura.automotive.can.session import CANSession

        session = CANSession(sample_can_messages)

        correlations = CorrelationAnalyzer.find_correlated_messages(session, 0x280, threshold=0.0)

        # Should not include 0x280 itself
        assert 0x280 not in correlations


class TestCorrelationEdgeCases:
    """Tests for edge cases in correlation analysis."""

    def test_single_sample_returns_zero(self):
        """Test that single sample returns zero correlation."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        msg1 = CANMessage(arbitration_id=0x100, timestamp=1.0, data=bytes(8))
        msg2 = CANMessage(arbitration_id=0x200, timestamp=1.0, data=bytes(8))

        messages1.append(msg1)
        messages2.append(msg2)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)
        assert correlation == 0.0

    def test_mismatched_message_counts(self):
        """Test correlation with different numbers of messages."""
        messages1 = CANMessageList()
        messages2 = CANMessageList()

        # 100 messages in first list
        for i in range(100):
            data = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data)
            messages1.append(msg)

        # Only 50 messages in second list
        for i in range(50):
            data = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg = CANMessage(arbitration_id=0x200, timestamp=i * 0.01, data=data)
            messages2.append(msg)

        correlation = CorrelationAnalyzer.correlate_bytes(messages1, 0, messages2, 0)

        # Should use truncated length (50)
        assert correlation != 0.0  # Should still compute correlation
