"""Unit tests for protocol replay validation framework.

This module tests the replay validation framework including:
- Serial interface (mocked)
- SocketCAN interface (mocked)
- UDP/TCP interfaces (mocked)
- Checksum validation
- Timing validation
- Error handling
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if pyserial not installed
pytest.importorskip("serial")

from oscura.validation.replay import (
    ProtocolSpec,
    ReplayConfig,
    ReplayValidator,
    ValidationResult,
)


class TestProtocolSpec:
    """Test ProtocolSpec dataclass."""

    def test_protocol_spec_defaults(self) -> None:
        """Test ProtocolSpec with default values."""
        spec = ProtocolSpec(name="TestProtocol")

        assert spec.name == "TestProtocol"
        assert spec.checksum_algorithm == "none"
        assert spec.checksum_position == -1
        assert spec.expected_response_time == 0.1
        assert spec.timing_tolerance == 0.2
        assert spec.require_response is True
        assert spec.message_format == ""

    def test_protocol_spec_custom_values(self) -> None:
        """Test ProtocolSpec with custom values."""
        spec = ProtocolSpec(
            name="UART",
            checksum_algorithm="crc16",
            checksum_position=7,
            expected_response_time=0.05,
            timing_tolerance=0.1,
            require_response=False,
            message_format="<header:1><data:6><crc:1>",
        )

        assert spec.name == "UART"
        assert spec.checksum_algorithm == "crc16"
        assert spec.checksum_position == 7
        assert spec.expected_response_time == 0.05
        assert spec.timing_tolerance == 0.1
        assert spec.require_response is False
        assert spec.message_format == "<header:1><data:6><crc:1>"


class TestReplayConfig:
    """Test ReplayConfig dataclass."""

    def test_replay_config_serial(self) -> None:
        """Test ReplayConfig for serial interface."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0", baud_rate=9600)

        assert config.interface == "serial"
        assert config.port == "/dev/ttyUSB0"
        assert config.baud_rate == 9600
        assert config.timeout == 1.0
        assert config.validate_checksums is True
        assert config.validate_timing is True
        assert config.max_retries == 3

    def test_replay_config_socketcan(self) -> None:
        """Test ReplayConfig for SocketCAN interface."""
        config = ReplayConfig(interface="socketcan", port="can0")

        assert config.interface == "socketcan"
        assert config.port == "can0"

    def test_replay_config_network(self) -> None:
        """Test ReplayConfig for network interface."""
        config = ReplayConfig(interface="udp", port=5000, host="192.168.1.1")

        assert config.interface == "udp"
        assert config.port == 5000
        assert config.host == "192.168.1.1"

    def test_replay_config_invalid_interface(self) -> None:
        """Test ReplayConfig rejects invalid interface."""
        with pytest.raises(ValueError, match="Invalid interface"):
            ReplayConfig(interface="invalid", port="/dev/null")  # type: ignore[arg-type]

    def test_replay_config_invalid_timeout(self) -> None:
        """Test ReplayConfig rejects invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ReplayConfig(interface="serial", port="/dev/null", timeout=-1.0)

    def test_replay_config_invalid_max_retries(self) -> None:
        """Test ReplayConfig rejects invalid max_retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReplayConfig(interface="serial", port="/dev/null", max_retries=-1)

    def test_replay_config_invalid_baud_rate(self) -> None:
        """Test ReplayConfig rejects invalid baud_rate."""
        with pytest.raises(ValueError, match="baud_rate must be positive"):
            ReplayConfig(interface="serial", port="/dev/null", baud_rate=0)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_success(self) -> None:
        """Test ValidationResult with successful validation."""
        result = ValidationResult(
            success=True,
            messages_sent=10,
            messages_received=10,
            checksum_valid=10,
            checksum_invalid=0,
            timing_valid=10,
            timing_invalid=0,
        )

        assert result.success is True
        assert result.messages_sent == 10
        assert result.messages_received == 10
        assert result.checksum_success_rate == 1.0
        assert result.timing_success_rate == 1.0
        assert result.response_rate == 1.0

    def test_validation_result_partial_success(self) -> None:
        """Test ValidationResult with partial success."""
        result = ValidationResult(
            success=False,
            messages_sent=10,
            messages_received=8,
            checksum_valid=6,
            checksum_invalid=2,
            timing_valid=7,
            timing_invalid=1,
            errors=["Message 5: timeout", "Message 9: timeout"],
        )

        assert result.success is False
        assert result.checksum_success_rate == 0.75  # 6/8
        assert result.timing_success_rate == pytest.approx(0.875)  # 7/8
        assert result.response_rate == 0.8  # 8/10
        assert len(result.errors) == 2

    def test_validation_result_zero_messages(self) -> None:
        """Test ValidationResult with zero messages."""
        result = ValidationResult(
            success=False,
            messages_sent=0,
            messages_received=0,
            checksum_valid=0,
            checksum_invalid=0,
            timing_valid=0,
            timing_invalid=0,
        )

        assert result.checksum_success_rate == 0.0
        assert result.timing_success_rate == 0.0
        assert result.response_rate == 0.0


class TestReplayValidator:
    """Test ReplayValidator class."""

    def test_validator_initialization(self) -> None:
        """Test validator initializes correctly."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        assert validator.config == config
        assert validator._connection is None
        assert validator._is_connected is False

    def test_connect_serial(self) -> None:
        """Test connecting to serial interface."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0", baud_rate=115200)
        validator = ReplayValidator(config)

        # Create mock serial module
        mock_serial_module = MagicMock()
        mock_serial_instance = MagicMock()
        mock_serial_module.Serial.return_value = mock_serial_instance

        # Patch sys.modules to inject our mock
        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            validator.connect()

            mock_serial_module.Serial.assert_called_once_with(
                port="/dev/ttyUSB0", baudrate=115200, timeout=1.0
            )
            assert validator._is_connected is True

    def test_connect_serial_missing_pyserial(self) -> None:
        """Test serial connection fails gracefully without pyserial."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        with patch.dict("sys.modules", {"serial": None}):
            with pytest.raises(ImportError, match="pyserial is required"):
                validator.connect()

    def test_connect_serial_invalid_port_type(self) -> None:
        """Test serial connection rejects invalid port type."""
        config = ReplayConfig(interface="serial", port=123)  # type: ignore[arg-type]
        validator = ReplayValidator(config)

        with pytest.raises(ValueError, match="Serial port must be string"):
            validator.connect()

    @patch("can.interface.Bus")
    def test_connect_socketcan(self, mock_bus: Mock) -> None:
        """Test connecting to SocketCAN interface."""
        config = ReplayConfig(interface="socketcan", port="can0")
        validator = ReplayValidator(config)

        validator.connect()

        mock_bus.assert_called_once_with(
            channel="can0", interface="socketcan", receive_own_messages=False
        )
        assert validator._is_connected is True

    def test_connect_socketcan_missing_python_can(self) -> None:
        """Test SocketCAN connection fails gracefully without python-can."""
        config = ReplayConfig(interface="socketcan", port="can0")
        validator = ReplayValidator(config)

        with patch.dict("sys.modules", {"can": None}):
            with pytest.raises(ImportError, match="python-can is required"):
                validator.connect()

    def test_connect_socketcan_invalid_port_type(self) -> None:
        """Test SocketCAN connection rejects invalid port type."""
        config = ReplayConfig(interface="socketcan", port=123)  # type: ignore[arg-type]
        validator = ReplayValidator(config)

        with pytest.raises(ValueError, match="CAN interface must be string"):
            validator.connect()

    @patch("socket.socket")
    def test_connect_udp(self, mock_socket: Mock) -> None:
        """Test connecting to UDP socket."""
        config = ReplayConfig(interface="udp", port=5000, host="localhost")
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock

        validator.connect()

        mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_sock.settimeout.assert_called_once_with(1.0)
        assert validator._is_connected is True

    @patch("socket.socket")
    def test_connect_tcp(self, mock_socket: Mock) -> None:
        """Test connecting to TCP socket."""
        config = ReplayConfig(interface="tcp", port=5000, host="localhost")
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock

        validator.connect()

        mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock.connect.assert_called_once_with(("localhost", 5000))
        assert validator._is_connected is True

    def test_connect_network_invalid_port_type(self) -> None:
        """Test network connection rejects invalid port type."""
        config = ReplayConfig(interface="udp", port="invalid")  # type: ignore[arg-type]
        validator = ReplayValidator(config)

        with pytest.raises(ValueError, match="Network port must be integer"):
            validator.connect()

    @patch("serial.Serial")
    def test_close_serial(self, mock_serial: Mock) -> None:
        """Test closing serial connection."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn

        validator.connect()
        validator.close()

        mock_conn.close.assert_called_once()
        assert validator._is_connected is False
        assert validator._connection is None

    def test_close_when_not_connected(self) -> None:
        """Test closing when not connected does nothing."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        validator.close()  # Should not raise

        assert validator._is_connected is False

    @patch("serial.Serial")
    def test_context_manager(self, mock_serial: Mock) -> None:
        """Test validator as context manager."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn

        with ReplayValidator(config) as validator:
            assert validator._is_connected is True

        mock_conn.close.assert_called_once()

    @patch("serial.Serial")
    def test_validate_protocol_not_connected(self, mock_serial: Mock) -> None:
        """Test validate_protocol fails if not connected."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)
        spec = ProtocolSpec(name="Test")

        with pytest.raises(RuntimeError, match="Not connected"):
            validator.validate_protocol(spec, [b"\x01\x02\x03"])

    @patch("serial.Serial")
    def test_validate_protocol_serial_success(self, mock_serial: Mock) -> None:
        """Test successful protocol validation via serial."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=False, validate_timing=False
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.read.return_value = b"\x04\x05\x06"

        validator.connect()
        spec = ProtocolSpec(name="Test", require_response=True)
        result = validator.validate_protocol(spec, [b"\x01\x02\x03"])

        assert result.success is True
        assert result.messages_sent == 1
        assert result.messages_received == 1
        assert len(result.response_log) == 1
        assert result.response_log[0]["message"] == "010203"
        assert result.response_log[0]["response"] == "040506"

    @patch("serial.Serial")
    def test_validate_protocol_no_response(self, mock_serial: Mock) -> None:
        """Test protocol validation with no response."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.read.return_value = b""  # No response

        validator.connect()
        spec = ProtocolSpec(name="Test", require_response=True)
        result = validator.validate_protocol(spec, [b"\x01\x02\x03"])

        assert result.success is False
        assert result.messages_sent == 1
        assert result.messages_received == 0
        assert len(result.errors) == 1
        assert "No response received" in result.errors[0]

    @patch("serial.Serial")
    def test_validate_protocol_checksum_validation(self, mock_serial: Mock) -> None:
        """Test protocol validation with checksum checking."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=True, validate_timing=False
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn

        # Message with XOR checksum: [0x01, 0x02, 0x03] -> checksum = 0x00
        mock_conn.read.return_value = b"\x01\x02\x03\x00"

        validator.connect()
        spec = ProtocolSpec(
            name="Test", checksum_algorithm="xor", checksum_position=-1, require_response=True
        )
        result = validator.validate_protocol(spec, [b"\xff\xff"])

        assert result.messages_sent == 1
        assert result.messages_received == 1
        assert result.checksum_valid == 1
        assert result.checksum_invalid == 0

    @patch("serial.Serial")
    def test_validate_protocol_invalid_checksum(self, mock_serial: Mock) -> None:
        """Test protocol validation with invalid checksum."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=True, validate_timing=False
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn

        # Message with bad checksum
        mock_conn.read.return_value = b"\x01\x02\x03\xff"

        validator.connect()
        spec = ProtocolSpec(
            name="Test", checksum_algorithm="xor", checksum_position=-1, require_response=True
        )
        result = validator.validate_protocol(spec, [b"\xff\xff"])

        assert result.checksum_valid == 0
        assert result.checksum_invalid == 1
        assert any("Invalid checksum" in err for err in result.errors)

    @patch("serial.Serial")
    @patch("time.time")
    def test_validate_protocol_timing_validation(self, mock_time: Mock, mock_serial: Mock) -> None:
        """Test protocol validation with timing checking."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=False, validate_timing=True
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.read.return_value = b"\x04\x05\x06"

        # Mock time to simulate 0.1s response time
        mock_time.side_effect = [0.0, 0.1]  # send, receive

        validator.connect()
        spec = ProtocolSpec(
            name="Test",
            expected_response_time=0.1,
            timing_tolerance=0.2,
            require_response=True,
        )
        result = validator.validate_protocol(spec, [b"\x01\x02\x03"])

        assert result.timing_valid == 1
        assert result.timing_invalid == 0

    @patch("serial.Serial")
    @patch("time.time")
    def test_validate_protocol_timing_out_of_tolerance(
        self, mock_time: Mock, mock_serial: Mock
    ) -> None:
        """Test protocol validation with timing outside tolerance."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=False, validate_timing=True
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.read.return_value = b"\x04\x05\x06"

        # Mock time to simulate 0.3s response time (3x expected)
        mock_time.side_effect = [0.0, 0.3]

        validator.connect()
        spec = ProtocolSpec(
            name="Test",
            expected_response_time=0.1,
            timing_tolerance=0.2,
            require_response=True,
        )
        result = validator.validate_protocol(spec, [b"\x01\x02\x03"])

        assert result.timing_valid == 0
        assert result.timing_invalid == 1
        assert any("outside tolerance" in err for err in result.errors)

    @patch("serial.Serial")
    def test_validate_protocol_multiple_messages(self, mock_serial: Mock) -> None:
        """Test protocol validation with multiple messages."""
        config = ReplayConfig(
            interface="serial", port="/dev/ttyUSB0", validate_checksums=False, validate_timing=False
        )
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.read.side_effect = [b"\x04", b"\x05", b"\x06"]

        validator.connect()
        spec = ProtocolSpec(name="Test", require_response=True)
        result = validator.validate_protocol(spec, [b"\x01", b"\x02", b"\x03"])

        assert result.success is True
        assert result.messages_sent == 3
        assert result.messages_received == 3
        assert len(result.response_log) == 3

    @patch("serial.Serial")
    def test_validate_protocol_exception_handling(self, mock_serial: Mock) -> None:
        """Test protocol validation handles exceptions gracefully."""
        config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        validator = ReplayValidator(config)

        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.write.side_effect = OSError("Write failed")

        validator.connect()
        spec = ProtocolSpec(name="Test", require_response=False)
        result = validator.validate_protocol(spec, [b"\x01\x02\x03"])

        assert result.success is False
        assert len(result.errors) == 1
        assert "OSError" in result.errors[0]

    def test_calculate_checksum_xor(self) -> None:
        """Test XOR checksum calculation."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        checksum = validator._calculate_checksum(b"\x01\x02\x03", "xor")
        assert checksum == 0x00  # 0x01 ^ 0x02 ^ 0x03 = 0x00

    def test_calculate_checksum_sum(self) -> None:
        """Test SUM checksum calculation."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        checksum = validator._calculate_checksum(b"\x01\x02\x03", "sum")
        assert checksum == 0x06  # 0x01 + 0x02 + 0x03 = 0x06

    def test_calculate_checksum_crc8(self) -> None:
        """Test CRC-8 checksum calculation."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        checksum = validator._calculate_checksum(b"\x01\x02\x03", "crc8")
        assert isinstance(checksum, int)
        assert 0 <= checksum <= 0xFF

    def test_calculate_checksum_crc16(self) -> None:
        """Test CRC-16 checksum calculation."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        checksum = validator._calculate_checksum(b"\x01\x02\x03", "crc16")
        assert isinstance(checksum, int)
        assert 0 <= checksum <= 0xFF  # Returns low byte

    def test_calculate_checksum_crc32(self) -> None:
        """Test CRC-32 checksum calculation."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        checksum = validator._calculate_checksum(b"\x01\x02\x03", "crc32")
        assert isinstance(checksum, int)
        assert 0 <= checksum <= 0xFF  # Returns low byte

    def test_calculate_checksum_unsupported(self) -> None:
        """Test calculate_checksum rejects unsupported algorithm."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        with pytest.raises(ValueError, match="Unsupported checksum algorithm"):
            validator._calculate_checksum(b"\x01\x02\x03", "invalid")

    def test_validate_checksum_empty_message(self) -> None:
        """Test validate_checksum handles empty message."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)
        spec = ProtocolSpec(name="Test", checksum_algorithm="xor")

        result = validator._validate_checksum(b"", spec)
        assert result is False

    def test_validate_checksum_position_out_of_bounds(self) -> None:
        """Test validate_checksum handles out-of-bounds position."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)
        spec = ProtocolSpec(name="Test", checksum_algorithm="xor", checksum_position=10)

        result = validator._validate_checksum(b"\x01\x02\x03", spec)
        assert result is False

    def test_validate_timing_within_tolerance(self) -> None:
        """Test validate_timing accepts timing within tolerance."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        result = validator._validate_timing(0.0, 0.1, 0.1, 0.2)
        assert result is True

    def test_validate_timing_outside_tolerance(self) -> None:
        """Test validate_timing rejects timing outside tolerance."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        result = validator._validate_timing(0.0, 0.2, 0.1, 0.2)
        assert result is False

    def test_validate_timing_lower_bound(self) -> None:
        """Test validate_timing at lower tolerance bound."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        # expected=0.1, tolerance=0.2 -> lower_bound=0.08
        result = validator._validate_timing(0.0, 0.08, 0.1, 0.2)
        assert result is True

    def test_validate_timing_upper_bound(self) -> None:
        """Test validate_timing at upper tolerance bound."""
        config = ReplayConfig(interface="serial", port="/dev/null")
        validator = ReplayValidator(config)

        # expected=0.1, tolerance=0.2 -> upper_bound=0.12
        result = validator._validate_timing(0.0, 0.12, 0.1, 0.2)
        assert result is True

    @patch("can.interface.Bus")
    def test_send_socketcan(self, mock_bus: Mock) -> None:
        """Test sending message via SocketCAN."""
        config = ReplayConfig(interface="socketcan", port="can0")
        validator = ReplayValidator(config)

        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance

        # Mock CAN message response
        mock_response = MagicMock()
        mock_response.data = b"\x04\x05\x06"
        mock_bus_instance.recv.return_value = mock_response

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response == b"\x04\x05\x06"
        mock_bus_instance.send.assert_called_once()
        mock_bus_instance.recv.assert_called_once_with(timeout=1.0)

    @patch("can.interface.Bus")
    def test_send_socketcan_timeout(self, mock_bus: Mock) -> None:
        """Test SocketCAN timeout returns None."""
        config = ReplayConfig(interface="socketcan", port="can0")
        validator = ReplayValidator(config)

        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        mock_bus_instance.recv.return_value = None

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response is None

    @patch("socket.socket")
    def test_send_udp(self, mock_socket: Mock) -> None:
        """Test sending message via UDP."""
        config = ReplayConfig(interface="udp", port=5000, host="localhost")
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recvfrom.return_value = (b"\x04\x05\x06", ("localhost", 5000))

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response == b"\x04\x05\x06"
        mock_sock.sendto.assert_called_once_with(b"\x01\x02\x03", ("localhost", 5000))

    @patch("socket.socket")
    def test_send_udp_timeout(self, mock_socket: Mock) -> None:
        """Test UDP timeout returns None."""
        config = ReplayConfig(interface="udp", port=5000)
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recvfrom.side_effect = TimeoutError()

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response is None

    @patch("socket.socket")
    def test_send_tcp(self, mock_socket: Mock) -> None:
        """Test sending message via TCP."""
        config = ReplayConfig(interface="tcp", port=5000, host="localhost")
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.return_value = b"\x04\x05\x06"

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response == b"\x04\x05\x06"
        mock_sock.sendall.assert_called_once_with(b"\x01\x02\x03")

    @patch("socket.socket")
    def test_send_tcp_timeout(self, mock_socket: Mock) -> None:
        """Test TCP timeout returns None."""
        config = ReplayConfig(interface="tcp", port=5000)
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = TimeoutError()

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response is None

    @patch("socket.socket")
    def test_send_tcp_empty_response(self, mock_socket: Mock) -> None:
        """Test TCP empty response returns None."""
        config = ReplayConfig(interface="tcp", port=5000)
        validator = ReplayValidator(config)

        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.return_value = b""

        validator.connect()
        response = validator._send_message(b"\x01\x02\x03")

        assert response is None
