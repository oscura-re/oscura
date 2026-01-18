"""Tests for CANSession class."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import pytest

from oscura.automotive.can.session import CANSession


class TestCANSession:
    """Tests for CANSession class."""

    def test_create_from_messages(self, sample_can_messages):
        """Test creating session from message list."""
        session = CANSession(messages=sample_can_messages)

        assert len(session) > 0
        assert len(session.unique_ids()) > 0

    def test_inventory(self, sample_can_messages):
        """Test message inventory generation."""
        session = CANSession(messages=sample_can_messages)
        inventory = session.inventory()

        # Should have entries for each unique ID
        assert len(inventory) == len(sample_can_messages.unique_ids())

        # Check columns exist
        assert "arbitration_id" in inventory.columns
        assert "count" in inventory.columns
        assert "frequency_hz" in inventory.columns
        assert "period_ms" in inventory.columns

    def test_message_wrapper(self, sample_can_messages):
        """Test getting message wrapper."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        assert msg.arbitration_id == 0x280

    def test_message_not_found(self, sample_can_messages):
        """Test getting non-existent message."""
        session = CANSession(messages=sample_can_messages)

        with pytest.raises(ValueError, match="No messages found"):
            session.message(0xFFF)

    def test_filter_by_ids(self, sample_can_messages):
        """Test filtering by arbitration IDs."""
        session = CANSession(messages=sample_can_messages)
        filtered = session.filter(arbitration_ids=[0x280, 0x300])

        assert len(filtered.unique_ids()) == 2
        assert 0x280 in filtered.unique_ids()
        assert 0x300 in filtered.unique_ids()

    def test_filter_by_time_range(self, sample_can_messages):
        """Test filtering by time range."""
        session = CANSession(messages=sample_can_messages)
        filtered = session.filter(time_range=(0.5, 0.8))

        # Should only include messages in time range
        start, end = filtered.time_range()
        assert start >= 0.5
        assert end <= 0.8

    def test_analyze_message_caching(self, sample_can_messages):
        """Test that message analysis is cached."""
        session = CANSession(messages=sample_can_messages)

        # First analysis
        analysis1 = session.analyze_message(0x280)

        # Second analysis (should use cache)
        analysis2 = session.analyze_message(0x280)

        # Should be same object (cached)
        assert analysis1 is analysis2

        # Force refresh
        analysis3 = session.analyze_message(0x280, force_refresh=True)
        assert analysis3 is not analysis1


@pytest.mark.unit
class TestMessageWrapper:
    """Tests for CANMessageWrapper class."""

    def test_analyze(self, sample_can_messages):
        """Test analyzing a message."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        analysis = msg.analyze()

        assert analysis.arbitration_id == 0x280
        assert analysis.message_count > 0
        assert len(analysis.byte_analyses) > 0

    def test_test_hypothesis_valid(self, sample_can_messages):
        """Test hypothesis testing with valid hypothesis."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        # Test hypothesis for RPM signal (bytes 2-3, scale 0.25)
        result = msg.test_hypothesis(
            signal_name="rpm",
            start_byte=2,
            bit_length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            expected_min=0,
            expected_max=10000,
        )

        assert len(result.values) > 0
        assert result.min_value >= 0
        assert result.max_value <= 10000
        # RPM should be in range 800-2000 based on test data
        assert 700 <= result.min_value <= 900
        assert 1900 <= result.max_value <= 2100

    def test_test_hypothesis_invalid(self, sample_can_messages):
        """Test hypothesis testing with invalid hypothesis."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        # Test with wrong byte position
        result = msg.test_hypothesis(
            signal_name="bad_signal",
            start_byte=0,  # Wrong byte (constant)
            bit_length=8,
            scale=1.0,
        )

        # Should detect constant values
        assert result.std == pytest.approx(0.0)
        assert result.confidence < 1.0

    def test_document_signal(self, sample_can_messages):
        """Test documenting a signal."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        msg.document_signal(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            comment="Engine RPM",
        )

        documented = msg.get_documented_signals()
        assert "rpm" in documented
        assert documented["rpm"].name == "rpm"
        assert documented["rpm"].start_bit == 16

    def test_decode_signals(self, sample_can_messages):
        """Test decoding documented signals."""
        session = CANSession(messages=sample_can_messages)
        msg = session.message(0x280)

        msg.document_signal(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
        )

        decoded = msg.decode_signals()

        assert len(decoded) > 0
        # Should have one decoded signal per message
        assert len(decoded) == len(sample_can_messages.filter_by_id(0x280))

        # Check first decoded signal
        sig = decoded[0]
        assert sig.name == "rpm"
        assert sig.unit == "rpm"
        assert sig.value > 0  # RPM should be positive
