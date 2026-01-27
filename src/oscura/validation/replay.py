"""Protocol replay validation framework.

This module provides a framework for validating reverse-engineered protocols by
replaying messages to target devices and comparing expected vs actual responses.

Supports multiple interfaces:
- Serial ports (UART, RS-232, RS-485)
- SocketCAN (Controller Area Network)
- UDP/TCP sockets (network protocols)

Example:
    >>> from oscura.validation.replay import ReplayConfig, ReplayValidator
    >>> config = ReplayConfig(
    ...     interface="serial",
    ...     port="/dev/ttyUSB0",
    ...     baud_rate=115200,
    ...     timeout=1.0
    ... )
    >>> validator = ReplayValidator(config)
    >>> spec = ProtocolSpec(
    ...     name="MyProtocol",
    ...     checksum_algorithm="crc16",
    ...     expected_response_time=0.1
    ... )
    >>> test_messages = [b"\\x01\\x02\\x03\\x04", b"\\x05\\x06\\x07\\x08"]
    >>> result = validator.validate_protocol(spec, test_messages)
    >>> print(f"Success: {result.success}, Messages: {result.messages_sent}")
    Success: True, Messages: 2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from oscura.utils.serial import connect_serial_port

if TYPE_CHECKING:
    import socket


class CANBusProtocol(Protocol):
    """Protocol for python-can Bus interface."""

    def send(self, msg: Any) -> None:
        """Send a CAN message."""
        ...

    def recv(self, timeout: float | None = None) -> Any:
        """Receive a CAN message."""
        ...


@dataclass
class ProtocolSpec:
    """Protocol specification for validation.

    Attributes:
        name: Protocol name for identification.
        checksum_algorithm: Checksum algorithm ("crc8", "crc16", "crc32", "xor", "sum").
        checksum_position: Byte position of checksum in message (-1 for last byte).
        expected_response_time: Expected response time in seconds.
        timing_tolerance: Timing tolerance as fraction (0.1 = 10% tolerance).
        require_response: Whether response is required for each message.
        message_format: Message format description (optional).
    """

    name: str
    checksum_algorithm: str = "none"
    checksum_position: int = -1
    expected_response_time: float = 0.1
    timing_tolerance: float = 0.2
    require_response: bool = True
    message_format: str = ""


@dataclass
class ReplayConfig:
    """Configuration for replay validation.

    Attributes:
        interface: Interface type ("serial", "socketcan", "udp", "tcp").
        port: Port identifier ("/dev/ttyUSB0" for serial, "can0" for CAN, port number for UDP/TCP).
        baud_rate: Baud rate for serial interface (default: 115200).
        timeout: Timeout in seconds for receiving responses (default: 1.0).
        validate_checksums: Enable checksum validation (default: True).
        validate_timing: Enable timing validation (default: True).
        max_retries: Maximum number of retries for failed sends (default: 3).
        host: Host address for UDP/TCP (default: "localhost").
    """

    interface: Literal["serial", "socketcan", "udp", "tcp"]
    port: str | int
    baud_rate: int = 115200
    timeout: float = 1.0
    validate_checksums: bool = True
    validate_timing: bool = True
    max_retries: int = 3
    host: str = "localhost"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.interface not in ("serial", "socketcan", "udp", "tcp"):
            raise ValueError(
                f"Invalid interface: {self.interface}. "
                "Must be 'serial', 'socketcan', 'udp', or 'tcp'"
            )
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.baud_rate <= 0:
            raise ValueError(f"baud_rate must be positive, got {self.baud_rate}")


@dataclass
class ValidationResult:
    """Results from protocol validation.

    Attributes:
        success: Overall validation success status.
        messages_sent: Number of messages successfully sent.
        messages_received: Number of responses received.
        checksum_valid: Number of valid checksums.
        checksum_invalid: Number of invalid checksums.
        timing_valid: Number of responses with valid timing.
        timing_invalid: Number of responses with invalid timing.
        errors: List of error messages encountered.
        response_log: Detailed log of message/response pairs.
    """

    success: bool
    messages_sent: int
    messages_received: int
    checksum_valid: int
    checksum_invalid: int
    timing_valid: int
    timing_invalid: int
    errors: list[str] = field(default_factory=list)
    response_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def checksum_success_rate(self) -> float:
        """Calculate checksum success rate.

        Returns:
            Fraction of checksums that were valid (0.0-1.0).
        """
        total = self.checksum_valid + self.checksum_invalid
        return self.checksum_valid / total if total > 0 else 0.0

    @property
    def timing_success_rate(self) -> float:
        """Calculate timing success rate.

        Returns:
            Fraction of responses with valid timing (0.0-1.0).
        """
        total = self.timing_valid + self.timing_invalid
        return self.timing_valid / total if total > 0 else 0.0

    @property
    def response_rate(self) -> float:
        """Calculate response rate.

        Returns:
            Fraction of messages that received responses (0.0-1.0).
        """
        return self.messages_received / self.messages_sent if self.messages_sent > 0 else 0.0


def _checksum_xor(data: bytes) -> int:
    """Calculate XOR checksum.

    Args:
        data: Data bytes to checksum.

    Returns:
        XOR checksum (single byte).
    """
    result = 0
    for byte in data:
        result ^= byte
    return result


def _checksum_sum(data: bytes) -> int:
    """Calculate sum checksum.

    Args:
        data: Data bytes to checksum.

    Returns:
        Sum checksum (single byte, truncated).
    """
    return sum(data) & 0xFF


def _checksum_crc8(data: bytes) -> int:
    """Calculate CRC-8 checksum (SAE J1850).

    Args:
        data: Data bytes to checksum.

    Returns:
        CRC-8 checksum.
    """
    crc = 0xFF
    poly = 0x1D
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFF
    return crc ^ 0xFF


def _checksum_crc16(data: bytes) -> int:
    """Calculate CRC-16 checksum (CCITT).

    Args:
        data: Data bytes to checksum.

    Returns:
        CRC-16 checksum low byte.
    """
    crc = 0xFFFF
    poly = 0x1021
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc & 0xFF


def _init_validation_counters() -> dict[str, int]:
    """Initialize validation counters.

    Returns:
        Dictionary with zero-initialized counters.
    """
    return {
        "messages_sent": 0,
        "messages_received": 0,
        "checksum_valid": 0,
        "checksum_invalid": 0,
        "timing_valid": 0,
        "timing_invalid": 0,
    }


def _create_log_entry(index: int, message: bytes, send_time: float) -> dict[str, Any]:
    """Create base log entry for message.

    Args:
        index: Message index.
        message: Message bytes.
        send_time: Send timestamp.

    Returns:
        Log entry dictionary.
    """
    return {
        "message_index": index,
        "message": message.hex(),
        "send_time": send_time,
    }


def _update_log_with_response(
    log_entry: dict[str, Any], response: bytes, recv_time: float, send_time: float
) -> None:
    """Update log entry with response data.

    Args:
        log_entry: Log entry to update.
        response: Response bytes.
        recv_time: Receive timestamp.
        send_time: Send timestamp.
    """
    log_entry["response"] = response.hex()
    log_entry["recv_time"] = recv_time
    log_entry["response_time"] = recv_time - send_time


def _validate_response(
    validator: ReplayValidator,
    response: bytes,
    spec: ProtocolSpec,
    index: int,
    recv_time: float,
    send_time: float,
    counters: dict[str, int],
    errors: list[str],
    log_entry: dict[str, Any],
) -> None:
    """Validate response checksum and timing.

    Args:
        validator: ReplayValidator instance.
        response: Response bytes.
        spec: Protocol specification.
        index: Message index.
        recv_time: Receive timestamp.
        send_time: Send timestamp.
        counters: Validation counters.
        errors: Error list.
        log_entry: Log entry to update.
    """
    # Validate checksum if enabled
    if validator.config.validate_checksums and spec.checksum_algorithm != "none":
        if validator._validate_checksum(response, spec):
            counters["checksum_valid"] += 1
            log_entry["checksum_valid"] = True
        else:
            counters["checksum_invalid"] += 1
            log_entry["checksum_valid"] = False
            errors.append(f"Message {index}: Invalid checksum in response")

    # Validate timing if enabled
    if validator.config.validate_timing:
        response_time = recv_time - send_time
        if validator._validate_timing(
            send_time, recv_time, spec.expected_response_time, spec.timing_tolerance
        ):
            counters["timing_valid"] += 1
            log_entry["timing_valid"] = True
        else:
            counters["timing_invalid"] += 1
            log_entry["timing_valid"] = False
            errors.append(f"Message {index}: Response time {response_time:.3f}s outside tolerance")


def _compute_overall_success(
    errors: list[str], counters: dict[str, int], spec: ProtocolSpec
) -> bool:
    """Compute overall validation success.

    Args:
        errors: List of errors.
        counters: Validation counters.
        spec: Protocol specification.

    Returns:
        True if validation successful.
    """
    return (
        len(errors) == 0
        and counters["messages_sent"] > 0
        and (
            not spec.require_response or counters["messages_received"] == counters["messages_sent"]
        )
    )


def _checksum_crc32(data: bytes) -> int:
    """Calculate CRC-32 checksum (IEEE 802.3).

    Args:
        data: Data bytes to checksum.

    Returns:
        CRC-32 checksum low byte.
    """
    crc = 0xFFFFFFFF
    poly = 0xEDB88320
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return (~crc) & 0xFF


class ReplayValidator:
    """Validator for protocol replay testing.

    Sends test messages to target devices via serial, CAN, or network interfaces,
    captures responses, and validates checksums, timing, and state transitions.

    Example:
        >>> config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
        >>> validator = ReplayValidator(config)
        >>> spec = ProtocolSpec(name="UART", checksum_algorithm="xor")
        >>> result = validator.validate_protocol(spec, [b"\\x01\\x02\\x03"])
        >>> validator.close()
    """

    def __init__(self, config: ReplayConfig) -> None:
        """Initialize validator with interface configuration.

        Args:
            config: Replay configuration specifying interface and parameters.

        Raises:
            ImportError: If required library for interface is not installed.
            ValueError: If configuration is invalid.
        """
        self.config = config
        self._connection: Any = None
        self._is_connected = False

    def __enter__(self) -> ReplayValidator:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Establish connection to target device.

        Raises:
            ImportError: If required library is not installed.
            OSError: If connection fails.
        """
        if self._is_connected:
            return

        if self.config.interface == "serial":
            self._connect_serial()
        elif self.config.interface == "socketcan":
            self._connect_socketcan()
        elif self.config.interface in ("udp", "tcp"):
            self._connect_network()

        self._is_connected = True

    def _connect_serial(self) -> None:
        """Connect to serial port.

        Raises:
            ValueError: If port is not a string.
            ImportError: If pyserial is not installed.
            OSError: If serial port cannot be opened.
        """
        if not isinstance(self.config.port, str):
            raise ValueError("Serial port must be string")
        self._connection = connect_serial_port(
            port=self.config.port,
            baud_rate=self.config.baud_rate,
            timeout=self.config.timeout,
        )

    def _connect_socketcan(self) -> None:
        """Connect to SocketCAN interface.

        Raises:
            ImportError: If python-can is not installed.
            OSError: If CAN interface cannot be opened.
        """
        try:
            import can
        except ImportError as e:
            raise ImportError(
                "python-can is required for SocketCAN interface. "
                "Install with: pip install python-can"
            ) from e

        if not isinstance(self.config.port, str):
            raise ValueError(f"CAN interface must be string, got {type(self.config.port)}")

        self._connection = can.interface.Bus(
            channel=self.config.port, interface="socketcan", receive_own_messages=False
        )

    def _connect_network(self) -> None:
        """Connect to UDP/TCP socket.

        Raises:
            OSError: If socket cannot be created or connected.
        """
        import socket

        if not isinstance(self.config.port, int):
            raise ValueError(f"Network port must be integer, got {type(self.config.port)}")

        if self.config.interface == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout)
        else:  # tcp
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            sock.connect((self.config.host, self.config.port))

        self._connection = sock

    def close(self) -> None:
        """Close connection to target device."""
        if not self._is_connected or self._connection is None:
            return

        try:
            if hasattr(self._connection, "close"):
                self._connection.close()
            elif hasattr(self._connection, "shutdown"):
                self._connection.shutdown()
        finally:
            self._connection = None
            self._is_connected = False

    def validate_protocol(self, spec: ProtocolSpec, test_messages: list[bytes]) -> ValidationResult:
        """Send test messages and validate responses.

        Args:
            spec: Protocol specification with validation criteria.
            test_messages: List of test messages to send.

        Returns:
            Validation result with success status and detailed metrics.

        Raises:
            RuntimeError: If not connected to target device.

        Example:
            >>> config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
            >>> validator = ReplayValidator(config)
            >>> validator.connect()
            >>> spec = ProtocolSpec(name="Test", checksum_algorithm="xor")
            >>> result = validator.validate_protocol(spec, [b"\\x01\\x02\\x03"])
            >>> print(f"Success: {result.success}")
            >>> validator.close()
        """
        if not self._is_connected:
            raise RuntimeError("Not connected. Call connect() first.")

        counters = _init_validation_counters()
        errors: list[str] = []
        response_log: list[dict[str, Any]] = []

        for i, message in enumerate(test_messages):
            try:
                send_time, response, recv_time = self._send_and_measure(message)
                counters["messages_sent"] += 1

                log_entry = _create_log_entry(i, message, send_time)

                if response is not None:
                    counters["messages_received"] += 1
                    _update_log_with_response(log_entry, response, recv_time, send_time)
                    _validate_response(
                        self, response, spec, i, recv_time, send_time, counters, errors, log_entry
                    )
                else:
                    log_entry["response"] = None
                    if spec.require_response:
                        errors.append(f"Message {i}: No response received")

                response_log.append(log_entry)

            except Exception as e:
                errors.append(f"Message {i}: {type(e).__name__}: {e}")
                response_log.append({"message_index": i, "message": message.hex(), "error": str(e)})

        success = _compute_overall_success(errors, counters, spec)

        return ValidationResult(
            success=success,
            messages_sent=counters["messages_sent"],
            messages_received=counters["messages_received"],
            checksum_valid=counters["checksum_valid"],
            checksum_invalid=counters["checksum_invalid"],
            timing_valid=counters["timing_valid"],
            timing_invalid=counters["timing_invalid"],
            errors=errors,
            response_log=response_log,
        )

    def _send_and_measure(self, message: bytes) -> tuple[float, bytes | None, float]:
        """Send message and measure timing.

        Args:
            message: Message bytes to send.

        Returns:
            Tuple of (send_time, response, recv_time).
        """
        send_time = time.time()
        response = self._send_message(message)
        recv_time = time.time()
        return send_time, response, recv_time

    def _send_message(self, message: bytes) -> bytes | None:
        """Send message and capture response.

        Args:
            message: Message bytes to send.

        Returns:
            Response bytes, or None if no response received.

        Raises:
            OSError: If send/receive fails.
        """
        if self.config.interface == "serial":
            return self._send_serial(message)
        elif self.config.interface == "socketcan":
            return self._send_socketcan(message)
        elif self.config.interface in ("udp", "tcp"):
            return self._send_network(message)

        return None

    def _send_serial(self, message: bytes) -> bytes | None:
        """Send message via serial port.

        Args:
            message: Message bytes to send.

        Returns:
            Response bytes, or None if timeout.
        """
        import serial  # type: ignore[import-untyped]

        ser: serial.Serial = self._connection
        ser.reset_input_buffer()  # Clear any old data
        ser.write(message)
        ser.flush()

        # Read response with timeout
        response = ser.read(1024)  # Read up to 1KB
        return response if response else None

    def _send_socketcan(self, message: bytes) -> bytes | None:
        """Send message via SocketCAN.

        Args:
            message: Message bytes to send (max 8 bytes for CAN 2.0, 64 for CAN-FD).

        Returns:
            Response bytes, or None if timeout.
        """
        import can

        bus: CANBusProtocol = self._connection

        # Create CAN message with standard ID 0x123 (configurable in future)
        msg = can.Message(arbitration_id=0x123, data=message, is_extended_id=False)
        bus.send(msg)

        # Wait for response
        response_msg = bus.recv(timeout=self.config.timeout)
        return bytes(response_msg.data) if response_msg else None

    def _send_network(self, message: bytes) -> bytes | None:
        """Send message via UDP/TCP socket.

        Args:
            message: Message bytes to send.

        Returns:
            Response bytes, or None if timeout.
        """

        sock: socket.socket = self._connection

        if self.config.interface == "udp":
            sock.sendto(message, (self.config.host, self.config.port))
            try:
                response, _ = sock.recvfrom(1024)
                return response
            except TimeoutError:
                return None
        else:  # tcp
            sock.sendall(message)
            try:
                response = sock.recv(1024)
                return response if response else None
            except TimeoutError:
                return None

    def _validate_checksum(self, message: bytes, spec: ProtocolSpec) -> bool:
        """Verify checksum in received message.

        Args:
            message: Message bytes to validate.
            spec: Protocol specification with checksum configuration.

        Returns:
            True if checksum is valid, False otherwise.

        Example:
            >>> validator = ReplayValidator(ReplayConfig("serial", "/dev/null"))
            >>> spec = ProtocolSpec(name="Test", checksum_algorithm="xor")
            >>> validator._validate_checksum(b"\\x01\\x02\\x03", spec)
            True
        """
        if len(message) == 0:
            return False

        # Extract checksum from message
        checksum_pos = spec.checksum_position if spec.checksum_position >= 0 else len(message) - 1
        if checksum_pos >= len(message):
            return False

        received_checksum = message[checksum_pos]

        # Calculate expected checksum on data portion
        if checksum_pos == len(message) - 1:
            data = message[:-1]
        else:
            data = message[:checksum_pos] + message[checksum_pos + 1 :]

        expected_checksum = self._calculate_checksum(data, spec.checksum_algorithm)

        return received_checksum == expected_checksum

    def _calculate_checksum(self, data: bytes, algorithm: str) -> int:
        """Calculate checksum using specified algorithm.

        Args:
            data: Data bytes to checksum.
            algorithm: Checksum algorithm ("crc8", "crc16", "crc32", "xor", "sum").

        Returns:
            Calculated checksum value (truncated to single byte for most algorithms).

        Raises:
            ValueError: If algorithm is not supported.

        Example:
            >>> validator = ReplayValidator(ReplayConfig("serial", "/dev/null"))
            >>> validator._calculate_checksum(b"\\x01\\x02\\x03", "xor")
            0
        """
        checksum_functions = {
            "xor": _checksum_xor,
            "sum": _checksum_sum,
            "crc8": _checksum_crc8,
            "crc16": _checksum_crc16,
            "crc32": _checksum_crc32,
        }

        if algorithm not in checksum_functions:
            raise ValueError(
                f"Unsupported checksum algorithm: {algorithm}. "
                f"Supported: {', '.join(checksum_functions.keys())}"
            )

        return checksum_functions[algorithm](data)

    def _validate_timing(
        self, send_time: float, recv_time: float, expected: float, tolerance: float
    ) -> bool:
        """Verify response timing within tolerance.

        Args:
            send_time: Message send timestamp.
            recv_time: Response receive timestamp.
            expected: Expected response time in seconds.
            tolerance: Timing tolerance as fraction (0.1 = 10%).

        Returns:
            True if timing is within tolerance, False otherwise.

        Example:
            >>> validator = ReplayValidator(ReplayConfig("serial", "/dev/null"))
            >>> validator._validate_timing(0.0, 0.1, 0.1, 0.2)
            True
            >>> validator._validate_timing(0.0, 0.2, 0.1, 0.2)
            False
        """
        actual_time = recv_time - send_time
        lower_bound = expected * (1.0 - tolerance)
        upper_bound = expected * (1.0 + tolerance)
        # Add small epsilon to handle floating point precision errors
        eps = 1e-9
        return (lower_bound - eps) <= actual_time <= (upper_bound + eps)


__all__ = [
    "ProtocolSpec",
    "ReplayConfig",
    "ReplayValidator",
    "ValidationResult",
]
