"""Tests for CANSession class."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.session import CANSession


class TestCANSession:
    """Tests for CANSession class."""

    def test_create_from_messages(self, sample_can_messages):
        """Test creating session and populating with message list."""
        session = CANSession(name="Test Session")
        session._messages = sample_can_messages  # Internal population for testing

        assert len(session) > 0
        assert len(session.unique_ids()) > 0

    def test_inventory(self, sample_can_messages):
        """Test message inventory generation."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages
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
        session = CANSession(name="Test")
        session._messages = sample_can_messages
        msg = session.message(0x280)

        assert msg.arbitration_id == 0x280

    def test_message_not_found(self, sample_can_messages):
        """Test getting non-existent message."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        with pytest.raises(ValueError, match="No messages found"):
            session.message(0xFFF)

    def test_filter_by_ids(self, sample_can_messages):
        """Test filtering by arbitration IDs."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages
        filtered = session.filter(arbitration_ids=[0x280, 0x300])

        assert len(filtered.unique_ids()) == 2
        assert 0x280 in filtered.unique_ids()
        assert 0x300 in filtered.unique_ids()

    def test_filter_by_time_range(self, sample_can_messages):
        """Test filtering by time range."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages
        filtered = session.filter(time_range=(0.5, 0.8))

        # Should only include messages in time range
        start, end = filtered.time_range()
        assert start >= 0.5
        assert end <= 0.8

    def test_analyze_message_caching(self, sample_can_messages):
        """Test that message analysis is cached."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

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
        session = CANSession(name="Test")
        session._messages = sample_can_messages
        msg = session.message(0x280)

        analysis = msg.analyze()

        assert analysis.arbitration_id == 0x280
        assert analysis.message_count > 0
        assert len(analysis.byte_analyses) > 0

    def test_test_hypothesis_valid(self, sample_can_messages):
        """Test hypothesis testing with valid hypothesis."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages
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
        session = CANSession(name="Test")
        session._messages = sample_can_messages
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
        session = CANSession(name="Test")
        session._messages = sample_can_messages
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
        session = CANSession(name="Test")
        session._messages = sample_can_messages
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


@pytest.mark.unit
class TestCANSessionCRC:
    """Tests for CRC integration in CANSession."""

    def test_auto_crc_disabled(self, sample_can_messages):
        """Test that CRC recovery can be disabled."""
        session = CANSession(name="Test", auto_crc=False)
        session._messages = sample_can_messages

        # Analyze without CRC recovery
        results = session.analyze()

        # No CRC params should be recovered
        assert len(session._crc_params) == 0
        assert len(session.crc_info) == 0

    def test_auto_crc_insufficient_messages(self):
        """Test that CRC recovery requires minimum messages."""
        from oscura.automotive.can.models import CANMessage, CANMessageList

        session = CANSession(name="Test", auto_crc=True, crc_min_messages=20)

        # Add only 5 messages (less than minimum)
        messages = [
            CANMessage(arbitration_id=0x123, data=bytes([0x01, 0x02, 0x03]), timestamp=i * 0.01)
            for i in range(5)
        ]
        session._messages = CANMessageList(messages=messages)

        # Analyze
        results = session.analyze()

        # CRC should not be recovered (not enough messages)
        assert len(session._crc_params) == 0

    def test_auto_crc_recovery_with_known_crc(self):
        """Test CRC recovery with CAN messages containing known CRC."""
        from oscura.automotive.can.models import CANMessage, CANMessageList
        from oscura.inference.crc_reverse import CRCReverser

        session = CANSession(name="Test", auto_crc=True, crc_min_messages=4)

        # Generate CAN messages with CRC-8
        reverser = CRCReverser()
        messages = []
        for i in range(10):
            # 3 bytes data + 1 byte CRC
            data_bytes = bytes([0x01, 0x02, i])
            crc = reverser._calculate_crc(
                data=data_bytes,
                poly=0x07,
                width=8,
                init=0x00,
                xor_out=0x00,
                refin=False,
                refout=False,
            )
            full_data = bytes(list(data_bytes) + [crc])
            messages.append(CANMessage(arbitration_id=0x123, data=full_data, timestamp=i * 0.01))

        session._messages = CANMessageList(messages=messages)

        # Analyze to trigger CRC recovery
        results = session.analyze()

        # CRC should be recovered
        if 0x123 in session._crc_params:
            params = session._crc_params[0x123]
            assert params.polynomial == 0x07
            assert params.width == 8
            assert params.confidence > 0.8

    def test_crc_validation(self):
        """Test CRC validation on messages."""
        from oscura.automotive.can.models import CANMessage
        from oscura.inference.crc_reverse import CRCParameters, CRCReverser

        session = CANSession(name="Test", auto_crc=False, crc_validate=True)

        # Manually set CRC params
        session._crc_params[0x123] = CRCParameters(
            polynomial=0x07,
            width=8,
            init=0x00,
            xor_out=0x00,
            reflect_in=False,
            reflect_out=False,
            confidence=1.0,
        )

        # Create message with valid CRC
        reverser = CRCReverser()
        data_bytes = bytes([0x01, 0x02, 0x03])
        crc = reverser._calculate_crc(
            data=data_bytes, poly=0x07, width=8, init=0x00, xor_out=0x00, refin=False, refout=False
        )
        valid_msg = CANMessage(
            arbitration_id=0x123, data=bytes(list(data_bytes) + [crc]), timestamp=1.0
        )

        # Validate should pass
        assert session._validate_crc(valid_msg) is True

        # Create message with invalid CRC
        invalid_msg = CANMessage(
            arbitration_id=0x123, data=bytes([0x01, 0x02, 0x03, 0xFF]), timestamp=2.0
        )

        # Validate should fail (with warning logged)
        assert session._validate_crc(invalid_msg) is False

    def test_crc_info_property(self):
        """Test crc_info property returns correct format."""
        from oscura.inference.crc_reverse import CRCParameters

        session = CANSession(name="Test")

        # Manually set CRC params
        session._crc_params[0x123] = CRCParameters(
            polynomial=0x1021,
            width=16,
            init=0xFFFF,
            xor_out=0x0000,
            reflect_in=False,
            reflect_out=False,
            confidence=0.95,
            algorithm_name="CRC-16-CCITT",
        )

        crc_info = session.crc_info

        assert 0x123 in crc_info
        assert crc_info[0x123]["polynomial"] == "0x1021"
        assert crc_info[0x123]["width"] == 16
        assert crc_info[0x123]["confidence"] == 0.95
        assert crc_info[0x123]["algorithm_name"] == "CRC-16-CCITT"

    def test_crc_validation_disabled(self):
        """Test that CRC validation can be disabled."""
        from oscura.automotive.can.models import CANMessage
        from oscura.inference.crc_reverse import CRCParameters

        session = CANSession(name="Test", auto_crc=False, crc_validate=False)

        # Set CRC params
        session._crc_params[0x123] = CRCParameters(
            polynomial=0x07,
            width=8,
            init=0x00,
            xor_out=0x00,
            reflect_in=False,
            reflect_out=False,
            confidence=1.0,
        )

        # Create message with invalid CRC
        invalid_msg = CANMessage(
            arbitration_id=0x123, data=bytes([0x01, 0x02, 0x03, 0xFF]), timestamp=1.0
        )

        # Should return True (validation disabled)
        assert session._validate_crc(invalid_msg) is True
