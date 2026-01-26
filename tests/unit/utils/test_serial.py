"""Comprehensive tests for serial port utility module.

Tests serial port connection helpers.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from oscura.utils.serial import connect_serial_port

# =============================================================================
# Basic Connection Tests
# =============================================================================


def test_connect_serial_port_basic() -> None:
    """Test basic serial port connection."""
    mock_serial_module = MagicMock()
    mock_conn = MagicMock()
    mock_serial_module.Serial.return_value = mock_conn

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        conn = connect_serial_port("/dev/ttyUSB0", 115200)

        assert conn is mock_conn
        mock_serial_module.Serial.assert_called_once_with(
            port="/dev/ttyUSB0",
            baudrate=115200,
            timeout=1.0,
        )


def test_connect_serial_port_custom_timeout() -> None:
    """Test serial port connection with custom timeout."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("/dev/ttyUSB0", 9600, timeout=5.0)

        mock_serial_module.Serial.assert_called_once_with(
            port="/dev/ttyUSB0",
            baudrate=9600,
            timeout=5.0,
        )


def test_connect_serial_port_with_kwargs() -> None:
    """Test serial port connection with additional kwargs."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("/dev/ttyUSB0", 115200, timeout=2.0, bytesize=8, parity="N", stopbits=1)

        mock_serial_module.Serial.assert_called_once_with(
            port="/dev/ttyUSB0",
            baudrate=115200,
            timeout=2.0,
            bytesize=8,
            parity="N",
            stopbits=1,
        )


# =============================================================================
# Port Name Tests
# =============================================================================


def test_connect_serial_port_linux_device() -> None:
    """Test connection to Linux serial device."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("/dev/ttyUSB0", 115200)

        mock_serial_module.Serial.assert_called_once()
        assert mock_serial_module.Serial.call_args[1]["port"] == "/dev/ttyUSB0"


def test_connect_serial_port_windows_device() -> None:
    """Test connection to Windows serial port."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("COM3", 9600)

        mock_serial_module.Serial.assert_called_once()
        assert mock_serial_module.Serial.call_args[1]["port"] == "COM3"


def test_connect_serial_port_various_devices() -> None:
    """Test connection to various device names."""
    devices = [
        "/dev/ttyUSB0",
        "/dev/ttyACM0",
        "/dev/ttyS0",
        "COM1",
        "COM10",
        "/dev/cu.usbserial",
    ]

    for device in devices:
        mock_serial_module = MagicMock()
        mock_serial_module.Serial.return_value = MagicMock()

        with patch.dict(sys.modules, {"serial": mock_serial_module}):
            connect_serial_port(device, 115200)

            assert mock_serial_module.Serial.call_args[1]["port"] == device


# =============================================================================
# Baud Rate Tests
# =============================================================================


def test_connect_serial_port_common_baud_rates() -> None:
    """Test connection with common baud rates."""
    baud_rates = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

    for baud in baud_rates:
        mock_serial_module = MagicMock()
        mock_serial_module.Serial.return_value = MagicMock()

        with patch.dict(sys.modules, {"serial": mock_serial_module}):
            connect_serial_port("/dev/ttyUSB0", baud)

            assert mock_serial_module.Serial.call_args[1]["baudrate"] == baud


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_connect_serial_port_pyserial_not_installed() -> None:
    """Test error when pyserial is not installed."""
    # Simulate module not found by removing from sys.modules
    with patch.dict(sys.modules, {"serial": None}):
        with pytest.raises(ImportError, match="pyserial is required"):
            connect_serial_port("/dev/ttyUSB0", 115200)


def test_connect_serial_port_invalid_port_type() -> None:
    """Test error when port is not a string."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        with pytest.raises(ValueError, match="Serial port must be string"):
            connect_serial_port(123, 115200)  # type: ignore[arg-type]


def test_connect_serial_port_invalid_port_type_none() -> None:
    """Test error when port is None."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        with pytest.raises(ValueError, match="Serial port must be string"):
            connect_serial_port(None, 115200)  # type: ignore[arg-type]


def test_connect_serial_port_invalid_port_type_list() -> None:
    """Test error when port is a list."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        with pytest.raises(ValueError, match="Serial port must be string"):
            connect_serial_port(["/dev/ttyUSB0"], 115200)  # type: ignore[arg-type]


def test_connect_serial_port_os_error_propagates() -> None:
    """Test that OSError from serial port opening propagates."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.side_effect = OSError("Port not found")

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        with pytest.raises(OSError, match="Port not found"):
            connect_serial_port("/dev/nonexistent", 115200)


# =============================================================================
# Return Value Tests
# =============================================================================


def test_connect_serial_port_returns_connection() -> None:
    """Test that function returns the serial connection object."""
    mock_serial_module = MagicMock()
    mock_conn = MagicMock()
    mock_serial_module.Serial.return_value = mock_conn

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        result = connect_serial_port("/dev/ttyUSB0", 115200)

        assert result is mock_conn


def test_connect_serial_port_connection_has_expected_attributes() -> None:
    """Test that returned connection has expected serial attributes."""
    mock_serial_module = MagicMock()
    mock_conn = MagicMock()
    mock_conn.write = MagicMock()
    mock_conn.read = MagicMock()
    mock_conn.close = MagicMock()
    mock_serial_module.Serial.return_value = mock_conn

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        conn = connect_serial_port("/dev/ttyUSB0", 115200)

        assert hasattr(conn, "write")
        assert hasattr(conn, "read")
        assert hasattr(conn, "close")


# =============================================================================
# Edge Cases
# =============================================================================


def test_connect_serial_port_empty_string_port() -> None:
    """Test connection with empty string port."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        # Empty string is still a string, so validation passes
        # but serial.Serial will likely fail
        connect_serial_port("", 115200)

        assert mock_serial_module.Serial.call_args[1]["port"] == ""


def test_connect_serial_port_zero_timeout() -> None:
    """Test connection with zero timeout (non-blocking)."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("/dev/ttyUSB0", 115200, timeout=0)

        assert mock_serial_module.Serial.call_args[1]["timeout"] == 0


def test_connect_serial_port_negative_timeout() -> None:
    """Test connection with negative timeout."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        # Function doesn't validate timeout value - passes to pyserial
        connect_serial_port("/dev/ttyUSB0", 115200, timeout=-1.0)

        assert mock_serial_module.Serial.call_args[1]["timeout"] == -1.0


def test_connect_serial_port_very_high_baud_rate() -> None:
    """Test connection with very high baud rate."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port("/dev/ttyUSB0", 12000000)

        assert mock_serial_module.Serial.call_args[1]["baudrate"] == 12000000


# =============================================================================
# Integration Tests
# =============================================================================


def test_connect_serial_port_multiple_connections() -> None:
    """Test opening multiple serial connections."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        conn1 = connect_serial_port("/dev/ttyUSB0", 115200)
        conn2 = connect_serial_port("/dev/ttyUSB1", 9600)

        assert conn1 is not None
        assert conn2 is not None
        assert mock_serial_module.Serial.call_count == 2


def test_connect_serial_port_with_all_common_kwargs() -> None:
    """Test connection with all common serial parameters."""
    mock_serial_module = MagicMock()
    mock_serial_module.Serial.return_value = MagicMock()

    with patch.dict(sys.modules, {"serial": mock_serial_module}):
        connect_serial_port(
            port="/dev/ttyUSB0",
            baud_rate=115200,
            timeout=2.0,
            bytesize=8,
            parity="N",
            stopbits=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

        call_kwargs = mock_serial_module.Serial.call_args[1]
        assert call_kwargs["port"] == "/dev/ttyUSB0"
        assert call_kwargs["baudrate"] == 115200
        assert call_kwargs["timeout"] == 2.0
        assert call_kwargs["bytesize"] == 8
        assert call_kwargs["parity"] == "N"
        assert call_kwargs["stopbits"] == 1
        assert call_kwargs["xonxoff"] is False
        assert call_kwargs["rtscts"] is False
        assert call_kwargs["dsrdtr"] is False
