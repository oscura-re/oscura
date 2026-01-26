"""Tests for SocketCAN hardware acquisition source.

Tests comprehensive coverage of SocketCANSource with mocked CAN bus.
Since SocketCAN interfaces require Linux and physical/virtual CAN, all can.Bus calls are mocked.

Coverage targets:
- Connection management (interface initialization, close)
- Read operations (CAN message capture, conversion to DigitalTrace)
- Streaming operations (chunked message streaming)
- Error handling (import errors, interface errors, empty messages)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.hardware.acquisition.socketcan import SocketCANSource


class MockCANMessage:
    """Mock CAN message for testing."""

    def __init__(self, timestamp: float, arbitration_id: int, data: bytes) -> None:
        self.timestamp = timestamp
        self.arbitration_id = arbitration_id
        self.data = data


class TestSocketCANSource:
    """Test SocketCANSource CAN bus acquisition functionality."""

    def test_init_basic(self) -> None:
        """Test basic initialization without bus connection."""
        source = SocketCANSource("can0")
        assert source.interface == "can0"
        assert source.bitrate == 500000  # Default bitrate
        assert source.bus is None
        assert source._closed is False

    def test_init_with_bitrate(self) -> None:
        """Test initialization with custom bitrate."""
        source = SocketCANSource("can0", bitrate=250000)
        assert source.interface == "can0"
        assert source.bitrate == 250000

    def test_init_with_kwargs(self) -> None:
        """Test initialization with additional can.Bus options."""
        source = SocketCANSource("vcan0", bitrate=500000, receive_own_messages=True)
        assert source.interface == "vcan0"
        assert source.kwargs == {"receive_own_messages": True}

    def test_ensure_bus_success(self) -> None:
        """Test successful CAN bus initialization."""
        mock_bus = MagicMock()
        mock_can = MagicMock()
        mock_can.Bus.return_value = mock_bus

        with patch.dict("sys.modules", {"can": mock_can}):
            source = SocketCANSource("can0", bitrate=500000)
            source._ensure_bus()

            assert source.bus == mock_bus
            mock_can.Bus.assert_called_once_with(
                interface="socketcan",
                channel="can0",
                bitrate=500000,
            )

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    def test_ensure_bus_with_kwargs(self, mock_can: Mock) -> None:
        """Test bus initialization with additional kwargs."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        source = SocketCANSource("can0", bitrate=250000, receive_own_messages=True)
        source._ensure_bus()

        mock_can.Bus.assert_called_once_with(
            interface="socketcan",
            channel="can0",
            bitrate=250000,
            receive_own_messages=True,
        )

    def test_ensure_bus_import_error(self) -> None:
        """Test handling of missing python-can library."""
        source = SocketCANSource("can0")

        with patch(
            "oscura.hardware.acquisition.socketcan.can", side_effect=ImportError("can not found")
        ):
            with pytest.raises(ImportError, match="SocketCAN source requires python-can library"):
                source._ensure_bus()

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    def test_ensure_bus_os_error(self, mock_can: Mock) -> None:
        """Test handling of interface open failure."""
        mock_can.Bus.side_effect = OSError("Interface can0 does not exist")

        source = SocketCANSource("can0")

        with pytest.raises(OSError, match="Failed to open SocketCAN interface"):
            source._ensure_bus()

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    def test_ensure_bus_idempotent(self, mock_can: Mock) -> None:
        """Test that bus is only initialized once."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        source = SocketCANSource("can0")
        source._ensure_bus()
        source._ensure_bus()

        # Should only call once
        mock_can.Bus.assert_called_once()

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    @patch("oscura.hardware.acquisition.socketcan.time", create=True)
    def test_read_with_messages(self, mock_time: Mock, mock_can: Mock) -> None:
        """Test reading CAN messages successfully."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        # Mock received messages
        messages = [
            MockCANMessage(0.001, 0x123, b"\x01\x02\x03\x04\x05\x06\x07\x08"),
            MockCANMessage(0.002, 0x456, b"\x11\x12\x13\x14\x15\x16\x17\x18"),
            MockCANMessage(0.003, 0x789, b"\x21\x22\x23\x24\x25\x26\x27\x28"),
        ]

        mock_time.time.side_effect = [0.0, 0.001, 0.002, 0.003, 1.0]  # Duration = 1.0 second
        mock_bus.recv.side_effect = messages + [None]  # None triggers timeout

        source = SocketCANSource("can0")
        trace = source.read(duration=1.0)

        from oscura.core.types import DigitalTrace

        assert isinstance(trace, DigitalTrace)
        assert len(trace.data) > 0
        assert trace.metadata.sample_rate > 0
        assert "socketcan://can0" in trace.metadata.source_file
        assert "CAN can0" in trace.metadata.channel_name

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    @patch("oscura.hardware.acquisition.socketcan.time", create=True)
    def test_read_empty_messages(self, mock_time: Mock, mock_can: Mock) -> None:
        """Test reading with no messages received."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        mock_time.time.side_effect = [0.0, 1.0]  # Duration elapsed
        mock_bus.recv.return_value = None  # No messages

        source = SocketCANSource("can0")
        trace = source.read(duration=1.0)

        from oscura.core.types import DigitalTrace

        assert isinstance(trace, DigitalTrace)
        assert len(trace.data) == 0
        assert trace.metadata.sample_rate == 1.0

    def test_read_closed_source_error(self) -> None:
        """Test that reading fails if source is closed."""
        source = SocketCANSource("can0")
        source._closed = True

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    @patch("oscura.hardware.acquisition.socketcan.time", create=True)
    def test_stream_messages(self, mock_time: Mock, mock_can: Mock) -> None:
        """Test streaming CAN messages in chunks."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        # Create 2500 messages to yield 3 chunks (1000 each)
        messages = [
            MockCANMessage(i * 0.001, 0x100 + (i % 256), bytes([i % 256] * 8)) for i in range(2500)
        ]

        # Simulate time progression
        time_vals = [0.0] + [i * 0.001 for i in range(2500)] + [10.0]
        mock_time.time.side_effect = time_vals

        mock_bus.recv.side_effect = messages + [None]

        source = SocketCANSource("can0")
        chunks = list(source.stream(duration=10.0, chunk_size=1000))

        # Should have 3 chunks (2 full + 1 partial)
        assert len(chunks) == 3

        for chunk in chunks[:2]:
            from oscura.core.types import DigitalTrace

            assert isinstance(chunk, DigitalTrace)

    def test_stream_closed_source_error(self) -> None:
        """Test that streaming fails if source is closed."""
        source = SocketCANSource("can0")
        source._closed = True

        with pytest.raises(ValueError, match="Cannot stream from closed source"):
            list(source.stream())

    def test_close(self) -> None:
        """Test closing source."""
        source = SocketCANSource("can0")
        source.bus = MagicMock()

        source.close()

        source.bus.shutdown.assert_called_once()
        assert source.bus is None
        assert source._closed is True

    def test_close_already_closed(self) -> None:
        """Test closing already closed source."""
        source = SocketCANSource("can0")
        source.close()
        source.close()  # Should not raise

        assert source._closed is True

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    def test_context_manager_success(self, mock_can: Mock) -> None:
        """Test context manager usage."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        with SocketCANSource("vcan0", bitrate=250000) as source:
            assert source.interface == "vcan0"
            assert source._closed is False

        # Should be closed after exiting context
        assert source._closed is True

    def test_repr(self) -> None:
        """Test string representation."""
        source = SocketCANSource("can0", bitrate=500000)
        assert repr(source) == "SocketCANSource(interface='can0', bitrate=500000)"

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    @patch("oscura.hardware.acquisition.socketcan.time", create=True)
    def test_acquisition_time_metadata(self, mock_time: Mock, mock_can: Mock) -> None:
        """Test that acquisition time is captured in metadata."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        mock_time.time.side_effect = [0.0, 1.0]
        mock_bus.recv.return_value = None

        source = SocketCANSource("can0")

        before = datetime.now()
        trace = source.read(duration=1.0)
        after = datetime.now()

        assert before <= trace.metadata.acquisition_time <= after

    @patch("oscura.hardware.acquisition.socketcan.can", create=True)
    @patch("oscura.hardware.acquisition.socketcan.time", create=True)
    def test_can_id_conversion(self, mock_time: Mock, mock_can: Mock) -> None:
        """Test CAN ID to digital data conversion."""
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        # Create messages with specific CAN IDs
        messages = [
            MockCANMessage(0.001, 0x123, b"\x00"),
            MockCANMessage(0.002, 0x456, b"\x00"),
        ]

        mock_time.time.side_effect = [0.0, 0.001, 0.002, 1.0]
        mock_bus.recv.side_effect = messages + [None]

        source = SocketCANSource("can0")
        trace = source.read(duration=1.0)

        # Verify CAN IDs are stored in trace data
        assert len(trace.data) > 0
