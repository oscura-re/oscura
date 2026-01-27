"""Serial port connection utilities.

This module provides serial port connection helpers used across validation modules.
"""

from typing import Any


def connect_serial_port(
    port: str,
    baud_rate: int,
    timeout: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Connect to serial port with validation.

    Args:
        port: Serial port path (e.g., '/dev/ttyUSB0', 'COM3')
        baud_rate: Baud rate (e.g., 9600, 115200)
        timeout: Read timeout in seconds
        **kwargs: Additional arguments passed to serial.Serial

    Returns:
        Serial connection object

    Raises:
        ImportError: If pyserial is not installed
        ValueError: If port is not a string
        OSError: If serial port cannot be opened

    Example:
        >>> conn = connect_serial_port('/dev/ttyUSB0', 115200)
        >>> conn.write(b'test')
        >>> conn.close()
    """
    try:
        import serial  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "pyserial is required for serial interface. Install with: pip install pyserial"
        ) from e

    if not isinstance(port, str):
        raise ValueError(f"Serial port must be string, got {type(port)}")

    return serial.Serial(
        port=port,
        baudrate=baud_rate,
        timeout=timeout,
        **kwargs,
    )
