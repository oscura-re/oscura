"""Tests for CAN data models."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import struct

import pytest

from oscura.automotive.can.models import (
    CANMessage,
    CANMessageList,
    DecodedSignal,
    SignalDefinition,
)


class TestCANMessage:
    """Tests for CANMessage class."""

    def test_create_message(self):
        """Test creating a CAN message."""
        msg = CANMessage(
            arbitration_id=0x123,
            timestamp=1.5,
            data=bytes([0x01, 0x02, 0x03, 0x04]),
            is_extended=False,
        )

        assert msg.arbitration_id == 0x123
        assert msg.timestamp == 1.5
        assert msg.data == bytes([0x01, 0x02, 0x03, 0x04])
        assert msg.dlc == 4
        assert not msg.is_extended
        assert not msg.is_fd

    def test_message_repr(self):
        """Test message string representation."""
        msg = CANMessage(
            arbitration_id=0x280,
            timestamp=2.5,
            data=bytes([0xAA, 0xBB, 0xCC, 0xDD]),
        )

        repr_str = repr(msg)
        assert "0x280" in repr_str
        assert "2.500000" in repr_str
        assert "AABBCCDD" in repr_str


class TestCANMessageList:
    """Tests for CANMessageList class."""

    def test_create_empty_list(self):
        """Test creating empty message list."""
        msg_list = CANMessageList()
        assert len(msg_list) == 0

    def test_append_messages(self):
        """Test appending messages."""
        msg_list = CANMessageList()

        msg1 = CANMessage(0x123, 0.0, bytes([0x01, 0x02]))
        msg2 = CANMessage(0x280, 0.1, bytes([0x03, 0x04]))

        msg_list.append(msg1)
        msg_list.append(msg2)

        assert len(msg_list) == 2
        assert msg_list[0] == msg1
        assert msg_list[1] == msg2

    def test_filter_by_id(self):
        """Test filtering by arbitration ID."""
        msg_list = CANMessageList()

        msg_list.append(CANMessage(0x123, 0.0, bytes([0x01])))
        msg_list.append(CANMessage(0x280, 0.1, bytes([0x02])))
        msg_list.append(CANMessage(0x123, 0.2, bytes([0x03])))
        msg_list.append(CANMessage(0x400, 0.3, bytes([0x04])))

        filtered = msg_list.filter_by_id(0x123)
        assert len(filtered) == 2
        assert all(msg.arbitration_id == 0x123 for msg in filtered)

    def test_unique_ids(self):
        """Test getting unique IDs."""
        msg_list = CANMessageList()

        msg_list.append(CANMessage(0x123, 0.0, bytes([0x01])))
        msg_list.append(CANMessage(0x280, 0.1, bytes([0x02])))
        msg_list.append(CANMessage(0x123, 0.2, bytes([0x03])))

        unique_ids = msg_list.unique_ids()
        assert unique_ids == {0x123, 0x280}

    def test_time_range(self):
        """Test getting time range."""
        msg_list = CANMessageList()

        msg_list.append(CANMessage(0x123, 1.0, bytes([0x01])))
        msg_list.append(CANMessage(0x280, 3.5, bytes([0x02])))
        msg_list.append(CANMessage(0x123, 2.2, bytes([0x03])))

        start, end = msg_list.time_range()
        assert start == 1.0
        assert end == 3.5


class TestSignalDefinition:
    """Tests for SignalDefinition class."""

    def test_create_definition(self):
        """Test creating signal definition."""
        sig_def = SignalDefinition(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            offset=0.0,
            unit="rpm",
        )

        assert sig_def.name == "rpm"
        assert sig_def.start_bit == 16
        assert sig_def.length == 16
        assert sig_def.start_byte == 2

    def test_extract_raw_big_endian(self):
        """Test extracting raw value (big-endian)."""
        # Create a signal at bytes 2-3, big-endian uint16
        sig_def = SignalDefinition(
            name="test",
            start_bit=16,
            length=16,
            byte_order="big_endian",
        )

        # Data: bytes 2-3 contain 0x1234
        data = bytes([0x00, 0x00, 0x12, 0x34, 0x00, 0x00, 0x00, 0x00])
        raw_value = sig_def.extract_raw(data)

        assert raw_value == 0x1234

    def test_decode_with_scale(self):
        """Test decoding with scale factor."""
        sig_def = SignalDefinition(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
        )

        # RPM encoded as 8000 (raw) = 2000 RPM (scaled)
        raw_rpm = 8000
        data = bytearray(8)
        data[2:4] = struct.pack(">H", raw_rpm)

        decoded = sig_def.decode(bytes(data))
        assert decoded == pytest.approx(2000.0)

    def test_decode_with_offset(self):
        """Test decoding with offset."""
        sig_def = SignalDefinition(
            name="temp",
            start_bit=0,
            length=8,
            scale=1.0,
            offset=-40.0,  # Temperature offset
            unit="°C",
        )

        # Raw value 100 = 60°C (100 - 40)
        data = bytes([100, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        decoded = sig_def.decode(data)

        assert decoded == pytest.approx(60.0)


class TestDecodedSignal:
    """Tests for DecodedSignal class."""

    def test_create_decoded_signal(self):
        """Test creating decoded signal."""
        sig = DecodedSignal(
            name="rpm",
            value=2000.0,
            unit="rpm",
            timestamp=1.5,
            raw_value=8000,
        )

        assert sig.name == "rpm"
        assert sig.value == 2000.0
        assert sig.unit == "rpm"
        assert sig.timestamp == 1.5
        assert sig.raw_value == 8000

    def test_decoded_signal_repr(self):
        """Test decoded signal representation."""
        sig = DecodedSignal(
            name="rpm",
            value=2000.5,
            unit="rpm",
            timestamp=1.5,
        )

        repr_str = repr(sig)
        assert "rpm" in repr_str
        assert "2000.50" in repr_str
        assert "rpm" in repr_str
