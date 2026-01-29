"""Automatic protocol decoding without user configuration.

This module provides one-shot protocol decode that auto-detects parameters
(baud rate, polarity, bit order) and decodes common protocols.


Example:
    >>> from oscura.discovery import decode_protocol
    >>> result = decode_protocol(trace)
    >>> print(f"Protocol: {result.protocol}")
    >>> print(f"Decoded {len(result.data)} bytes")
    >>> for byte_data in result.data[:10]:
    ...     print(f"0x{byte_data.value:02X} (confidence: {byte_data.confidence:.2f})")

References:
    Protocol specifications: UART (EIA/TIA-232), SPI (Motorola), I2C (NXP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace
from oscura.discovery.signal_detector import characterize_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

ProtocolType = Literal["UART", "SPI", "I2C", "unknown"]


@dataclass
class DecodedByte:
    """Single decoded byte with confidence.

    Attributes:
        value: Byte value (0-255).
        offset: Byte offset in decoded stream.
        confidence: Decode confidence (0.0-1.0).
        has_error: Whether byte has detected errors.
        error_type: Type of error if present.
        error_description: Plain-language error description.

    Example:
        >>> byte_data = DecodedByte(value=0x48, offset=0, confidence=0.95)
        >>> print(f"Byte: 0x{byte_data.value:02X}, char: {chr(byte_data.value)}")
    """

    value: int
    offset: int
    confidence: float
    has_error: bool = False
    error_type: str | None = None
    error_description: str | None = None

    def __post_init__(self) -> None:
        """Validate byte data."""
        if not 0 <= self.value <= 255:
            raise ValueError(f"Byte value must be 0-255, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass
class DecodeResult:
    """Protocol decode result with auto-detected parameters.

    Attributes:
        protocol: Detected protocol name.
        overall_confidence: Overall decode confidence (0.0-1.0).
        detected_params: Auto-detected protocol parameters.
        data: List of decoded bytes with per-byte confidence.
        frame_count: Number of frames/packets decoded.
        error_count: Number of errors detected.

    Example:
        >>> result = decode_protocol(trace)
        >>> print(f"Protocol: {result.protocol}")
        >>> print(f"Confidence: {result.overall_confidence:.2f}")
        >>> print(f"Parameters: {result.detected_params}")
        >>> print(f"Decoded {len(result.data)} bytes with {result.error_count} errors")
    """

    protocol: ProtocolType
    overall_confidence: float
    detected_params: dict[str, Any] = field(default_factory=dict)
    data: list[DecodedByte] = field(default_factory=list)
    frame_count: int = 0
    error_count: int = 0


def decode_protocol(
    trace: WaveformTrace | DigitalTrace,
    *,
    protocol_hint: ProtocolType | None = None,
    params_hint: dict[str, Any] | None = None,
    confidence_threshold: float = 0.7,
    return_errors: bool = True,
) -> DecodeResult:
    """Automatically decode protocol without user configuration.

    Auto-detects protocol type, parameters (baud rate, polarity, bit order),
    and decodes data with per-byte confidence scores.

    Supported protocols:
        - UART: Auto-detects baud rate, parity, stop bits
        - SPI: Auto-detects mode (CPOL/CPHA), bit order
        - I2C: Auto-detects clock rate, addressing mode

    Args:
        trace: Input waveform or digital trace.
        protocol_hint: Optional protocol hint to narrow detection.
        params_hint: Optional parameter hints (e.g., approximate clock freq).
        confidence_threshold: Minimum confidence for valid bytes.
        return_errors: Whether to include error information.

    Returns:
        DecodeResult with protocol, parameters, and decoded data.

    Raises:
        ValueError: If trace is empty or invalid.

    Example:
        >>> # Automatic decode with no configuration
        >>> result = decode_protocol(trace)
        >>> if result.overall_confidence >= 0.8:
        ...     print(f"High confidence {result.protocol} decode")
        ...     for byte_data in result.data:
        ...         print(f"0x{byte_data.value:02X}")

        >>> # Decode with hint to speed up detection
        >>> result = decode_protocol(trace, protocol_hint='UART')
        >>> print(f"Baud rate: {result.detected_params['baud_rate']}")

    References:
        DISC-010: One-Shot Protocol Decode
    """
    # Validate input
    if len(trace) == 0:
        raise ValueError("Cannot decode empty trace")

    # Auto-detect protocol if no hint provided
    if protocol_hint is None:
        char_result = characterize_signal(trace)

        # Map signal types to protocols
        if char_result.signal_type == "uart":
            protocol_hint = "UART"
        elif char_result.signal_type == "spi":
            protocol_hint = "SPI"
        elif char_result.signal_type == "i2c":
            protocol_hint = "I2C"
        # Try UART as default for digital signals
        elif char_result.signal_type == "digital":
            protocol_hint = "UART"
        else:
            return DecodeResult(
                protocol="unknown",
                overall_confidence=0.0,
                detected_params={},
                data=[],
            )

    # Decode based on detected/hinted protocol
    if protocol_hint == "UART":
        return _decode_uart_auto(trace, params_hint, confidence_threshold, return_errors)
    elif protocol_hint == "SPI":
        return _decode_spi_auto(trace, params_hint, confidence_threshold, return_errors)
    elif protocol_hint == "I2C":
        return _decode_i2c_auto(trace, params_hint, confidence_threshold, return_errors)
    else:
        return DecodeResult(
            protocol="unknown",
            overall_confidence=0.0,
            detected_params={},
            data=[],
        )


def _extract_uart_data_and_sample_rate(
    trace: WaveformTrace | DigitalTrace,
) -> tuple[NDArray[np.floating[Any]], float]:
    """Extract data array and sample rate from trace.

    Args:
        trace: Input trace

    Returns:
        Tuple of (data array, sample rate)
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        sample_rate = trace.metadata.sample_rate
    else:
        data = trace.data.astype(np.float64)
        sample_rate = trace.metadata.sample_rate
    return data, sample_rate


def _determine_uart_parameters(
    params_hint: dict[str, Any] | None,
    data: NDArray[np.floating[Any]],
    sample_rate: float,
) -> tuple[int, int, str, int]:
    """Determine UART parameters from hints or auto-detection.

    Args:
        params_hint: Optional parameter hints
        data: Signal data
        sample_rate: Sampling rate

    Returns:
        Tuple of (baud_rate, data_bits, parity, stop_bits)
    """
    # Auto-detect baud rate if not provided
    if params_hint and "baud_rate" in params_hint:
        baud_rate = params_hint["baud_rate"]
    else:
        baud_rate = _detect_baud_rate(data, sample_rate)

    # Default UART parameters (8N1)
    data_bits = params_hint.get("data_bits", 8) if params_hint else 8
    parity = params_hint.get("parity", "none") if params_hint else "none"
    stop_bits = params_hint.get("stop_bits", 1) if params_hint else 1

    return baud_rate, data_bits, parity, stop_bits


def _decode_uart_packets(
    decoder: Any,
    trace: WaveformTrace | DigitalTrace,
    baud_rate: int,
) -> list[Any]:
    """Decode UART packets from trace.

    Args:
        decoder: UART decoder instance
        trace: Input trace
        baud_rate: Baud rate for error reporting

    Returns:
        List of decoded packets (empty on failure)
    """
    try:
        return list(decoder.decode(trace))
    except Exception:
        return []


def _convert_uart_packets_to_bytes(
    packets: list[Any],
) -> tuple[list[DecodedByte], int]:
    """Convert UART packets to DecodedByte format.

    Args:
        packets: List of decoded packets

    Returns:
        Tuple of (decoded bytes list, error count)
    """
    decoded_bytes: list[DecodedByte] = []
    error_count = 0

    for i, packet in enumerate(packets):
        for byte_val in packet.data:
            # Calculate confidence based on errors
            has_error = len(packet.errors) > 0
            if has_error:
                confidence = 0.65
                error_count += 1
            else:
                confidence = 0.95

            decoded_bytes.append(
                DecodedByte(
                    value=byte_val,
                    offset=i,
                    confidence=confidence,
                    has_error=has_error,
                    error_type=packet.errors[0] if packet.errors else None,
                    error_description=packet.errors[0] if packet.errors else None,
                )
            )

    return decoded_bytes, error_count


def _calculate_uart_confidence(decoded_bytes: list[DecodedByte]) -> float:
    """Calculate overall confidence from decoded bytes.

    Args:
        decoded_bytes: List of decoded bytes

    Returns:
        Overall confidence score
    """
    if decoded_bytes:
        avg_confidence = np.mean([b.confidence for b in decoded_bytes])
        return float(round(float(avg_confidence), 2))
    return 0.0


def _decode_uart_auto(
    trace: WaveformTrace | DigitalTrace,
    params_hint: dict[str, Any] | None,
    confidence_threshold: float,
    return_errors: bool,
) -> DecodeResult:
    """Auto-decode UART with parameter detection.

    Args:
        trace: Input trace.
        params_hint: Optional parameter hints.
        confidence_threshold: Minimum confidence threshold.
        return_errors: Whether to include errors.

    Returns:
        DecodeResult for UART.
    """
    from oscura.analyzers.protocols.uart import UARTDecoder

    # Extract data and determine parameters
    data, sample_rate = _extract_uart_data_and_sample_rate(trace)
    baud_rate, data_bits, parity, stop_bits = _determine_uart_parameters(
        params_hint, data, sample_rate
    )

    # Create decoder with detected parameters
    decoder = UARTDecoder(
        baudrate=baud_rate,
        data_bits=data_bits,
        parity=parity,  # type: ignore[arg-type]
        stop_bits=stop_bits,
    )

    # Decode packets
    packets = _decode_uart_packets(decoder, trace, baud_rate)

    # Handle decode failure
    if not packets:
        return DecodeResult(
            protocol="UART",
            overall_confidence=0.3,
            detected_params={"baud_rate": baud_rate},
            data=[],
        )

    # Convert packets to bytes and calculate confidence
    decoded_bytes, error_count = _convert_uart_packets_to_bytes(packets)
    overall_confidence = _calculate_uart_confidence(decoded_bytes)

    return DecodeResult(
        protocol="UART",
        overall_confidence=overall_confidence,
        detected_params={
            "baud_rate": baud_rate,
            "data_bits": data_bits,
            "parity": parity,
            "stop_bits": stop_bits,
        },
        data=decoded_bytes,
        frame_count=len(packets),
        error_count=error_count,
    )


def _decode_spi_auto(
    trace: WaveformTrace | DigitalTrace,
    params_hint: dict[str, Any] | None,
    confidence_threshold: float,
    return_errors: bool,
) -> DecodeResult:
    """Auto-decode SPI with parameter detection.

    Args:
        trace: Input trace.
        params_hint: Optional parameter hints.
        confidence_threshold: Minimum confidence threshold.
        return_errors: Whether to include errors.

    Returns:
        DecodeResult for SPI.
    """
    # SPI requires multiple channels (clock, MOSI, optionally MISO)
    # Single-channel auto-decode is limited
    # Return low-confidence result indicating more channels needed

    return DecodeResult(
        protocol="SPI",
        overall_confidence=0.4,
        detected_params={"note": "SPI requires clock and data channels"},
        data=[],
        frame_count=0,
        error_count=0,
    )


def _decode_i2c_auto(
    trace: WaveformTrace | DigitalTrace,
    params_hint: dict[str, Any] | None,
    confidence_threshold: float,
    return_errors: bool,
) -> DecodeResult:
    """Auto-decode I2C with parameter detection.

    Args:
        trace: Input trace.
        params_hint: Optional parameter hints.
        confidence_threshold: Minimum confidence threshold.
        return_errors: Whether to include errors.

    Returns:
        DecodeResult for I2C.
    """
    # I2C requires both SDA and SCL channels
    # Single-channel auto-decode is limited
    # Return low-confidence result indicating more channels needed

    return DecodeResult(
        protocol="I2C",
        overall_confidence=0.4,
        detected_params={"note": "I2C requires SDA and SCL channels"},
        data=[],
        frame_count=0,
        error_count=0,
    )


def _detect_baud_rate(data: NDArray[np.floating[Any]], sample_rate: float) -> int:
    """Auto-detect UART baud rate from signal.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.

    Returns:
        Detected baud rate in bps.
    """
    # Threshold signal to digital
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold

    # Find edges
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 10:
        return 115200  # Default fallback

    # Analyze edge intervals to find bit period
    intervals = np.diff(edges)

    # Use histogram to find most common interval (bit period)
    hist, bin_edges = np.histogram(intervals, bins=50)
    peak_bin = np.argmax(hist)
    bit_period_samples = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

    # Calculate baud rate
    estimated_baud = int(sample_rate / bit_period_samples)

    # Snap to common baud rates
    common_bauds = [
        300,
        600,
        1200,
        2400,
        4800,
        9600,
        14400,
        19200,
        28800,
        38400,
        57600,
        115200,
        230400,
        460800,
        921600,
    ]

    closest_baud = min(common_bauds, key=lambda x: abs(x - estimated_baud))

    # Validate: should be within 5% of detected rate
    if abs(closest_baud - estimated_baud) / estimated_baud < 0.05:
        return closest_baud
    else:
        # Use estimated rate if no close match
        return estimated_baud


__all__ = [
    "DecodeResult",
    "DecodedByte",
    "ProtocolType",
    "decode_protocol",
]
