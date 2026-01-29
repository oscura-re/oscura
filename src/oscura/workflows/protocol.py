"""Protocol debug workflow.

This module implements auto-detect protocol decoding with error context.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('serial_capture.wfm')
    >>> result = osc.debug_protocol(trace)
    >>> print(f"Protocol: {result['protocol']}")
    >>> print(f"Errors: {len(result['errors'])}")

References:
    UART: TIA-232-F
    I2C: NXP UM10204
    SPI: Motorola SPI Block Guide
    CAN: ISO 11898
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oscura.core.exceptions import AnalysisError
from oscura.core.types import DigitalTrace, ProtocolPacket, WaveformTrace


def debug_protocol(
    trace: WaveformTrace | DigitalTrace,
    *,
    protocol: str | None = None,
    context_samples: int = 100,
    error_types: list[str] | None = None,
    decode_all: bool = False,
) -> dict[str, Any]:
    """Auto-detect and decode protocol with error context.

    Automatically detects protocol type (UART, SPI, I2C, CAN) if not specified,
    decodes packets, and highlights errors with surrounding context samples.

    Args:
        trace: Signal to decode.
        protocol: Protocol type override ('UART', 'SPI', 'I2C', 'CAN', 'auto').
        context_samples: Number of samples before/after errors.
        error_types: Error types to detect. If None, detects all.
        decode_all: If True, decode all packets. If False, focus on errors.

    Returns:
        Dictionary with protocol, baud_rate, packets, errors, config, statistics.

    Raises:
        AnalysisError: If protocol cannot be detected or decoded.

    Example:
        >>> result = osc.debug_protocol(trace)
        >>> print(f"Protocol: {result['protocol']}")

    References:
        sigrok Protocol Decoder API, UART: TIA-232-F, I2C: NXP UM10204
    """
    digital_trace = _to_digital(trace)
    protocol, config, confidence = _detect_or_use_protocol(protocol, trace)

    packets, errors = _decode_protocol(
        protocol, digital_trace, config, context_samples, error_types, trace
    )

    if not decode_all:
        packets = [p for p in packets if p.errors]

    statistics = _compute_protocol_statistics(packets, errors, confidence)

    return {
        "protocol": protocol,
        "baud_rate": config.get("baud_rate") or config.get("clock_rate"),
        "packets": packets,
        "errors": errors,
        "config": config,
        "statistics": statistics,
    }


def _detect_or_use_protocol(
    protocol: str | None, trace: WaveformTrace | DigitalTrace
) -> tuple[str, dict[str, Any], float]:
    """Detect protocol or use specified protocol.

    Args:
        protocol: Protocol name or None for auto-detection.
        trace: Input trace.

    Returns:
        Tuple of (protocol_name, config, confidence).
    """
    from oscura.inference.protocol import detect_protocol

    if protocol is None or protocol.lower() == "auto":
        detection = detect_protocol(trace, min_confidence=0.5, return_candidates=True)  # type: ignore[arg-type]
        return detection["protocol"], detection["config"], detection["confidence"]
    else:
        config = _get_default_protocol_config(protocol.upper())
        return protocol.upper(), config, 1.0


def _decode_protocol(
    protocol: str,
    digital_trace: DigitalTrace,
    config: dict[str, Any],
    context_samples: int,
    error_types: list[str] | None,
    trace: WaveformTrace | DigitalTrace,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]]]:
    """Decode protocol based on type.

    Args:
        protocol: Protocol type (UART, SPI, I2C, CAN).
        digital_trace: Digital trace to decode.
        config: Protocol configuration.
        context_samples: Context sample count.
        error_types: Error types to detect.
        trace: Original trace for context.

    Returns:
        Tuple of (packets, errors).

    Raises:
        AnalysisError: If protocol is unsupported.
    """
    if protocol == "UART":
        return _decode_uart(digital_trace, config, context_samples, error_types, trace)
    elif protocol == "SPI":
        return _decode_spi(digital_trace, config, context_samples, error_types, trace)
    elif protocol == "I2C":
        return _decode_i2c(digital_trace, config, context_samples, error_types, trace)
    elif protocol == "CAN":
        return _decode_can(digital_trace, config, context_samples, error_types, trace)
    else:
        raise AnalysisError(f"Unsupported protocol: {protocol}")


def _compute_protocol_statistics(
    packets: list[ProtocolPacket], errors: list[dict[str, Any]], confidence: float
) -> dict[str, float | int]:
    """Compute protocol decoding statistics.

    Args:
        packets: Decoded packets list.
        errors: Error list.
        confidence: Detection confidence.

    Returns:
        Dictionary with statistics.
    """
    total_packets = len(packets)
    error_count = len(errors)

    return {
        "total_packets": total_packets,
        "error_count": error_count,
        "error_rate": error_count / total_packets if total_packets > 0 else 0,
        "confidence": confidence,
    }


def _to_digital(trace: WaveformTrace | DigitalTrace) -> DigitalTrace:
    """Convert waveform trace to digital trace.

    Args:
        trace: Input trace.

    Returns:
        Digital trace.
    """
    if isinstance(trace, DigitalTrace):
        return trace

    data = trace.data
    threshold = (np.max(data) + np.min(data)) / 2
    digital_data = data > threshold

    return DigitalTrace(
        data=digital_data,
        metadata=trace.metadata,
    )


def _get_default_protocol_config(protocol: str) -> dict[str, Any]:
    """Get default configuration for a protocol.

    Args:
        protocol: Protocol name.

    Returns:
        Default configuration dictionary.
    """
    configs = {
        "UART": {
            "baud_rate": 115200,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1,
        },
        "SPI": {
            "clock_polarity": 0,
            "clock_phase": 0,
            "bit_order": "MSB",
        },
        "I2C": {
            "clock_rate": 100000,  # Standard mode
            "address_bits": 7,
        },
        "CAN": {
            "baud_rate": 500000,
            "sample_point": 0.75,
        },
    }
    return configs.get(protocol, {})  # type: ignore[return-value]


def _extract_context(
    trace: WaveformTrace | DigitalTrace,
    sample_idx: int,
    context_samples: int,
) -> WaveformTrace | DigitalTrace | None:
    """Extract context samples around a point.

    Args:
        trace: Original trace.
        sample_idx: Center sample index.
        context_samples: Number of samples before and after.

    Returns:
        Sub-trace with context, or None if invalid.
    """
    data = trace.data
    start = max(0, sample_idx - context_samples)
    end = min(len(data), sample_idx + context_samples)

    if end <= start:
        return None

    context_data = data[start:end]

    if isinstance(trace, WaveformTrace):
        return WaveformTrace(
            data=context_data,  # type: ignore[arg-type]
            metadata=trace.metadata,
        )
    else:
        return DigitalTrace(
            data=context_data,  # type: ignore[arg-type]
            metadata=trace.metadata,
        )


def _decode_uart(
    trace: DigitalTrace,
    config: dict[str, Any],
    context_samples: int,
    error_types: list[str] | None,
    original_trace: WaveformTrace | DigitalTrace,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]]]:
    """Decode UART protocol with error context.

    Args:
        trace: Digital trace to decode.
        config: UART configuration.
        context_samples: Context window size.
        error_types: Error types to detect.
        original_trace: Original trace for context extraction.

    Returns:
        Tuple of (packets, errors).
    """
    from oscura.analyzers.protocols.uart import UARTDecoder

    baud_rate = config.get("baud_rate", 0)
    data_bits = config.get("data_bits", 8)
    parity = config.get("parity", "none")
    stop_bits = config.get("stop_bits", 1)

    decoder = UARTDecoder(
        baudrate=baud_rate,
        data_bits=data_bits,
        parity=parity,
        stop_bits=stop_bits,
    )

    packets = list(decoder.decode(trace))
    errors: list[dict[str, Any]] = []

    sample_rate = trace.metadata.sample_rate

    for i, pkt in enumerate(packets):
        if pkt.errors:
            # Filter by error types if specified
            relevant_errors = pkt.errors
            if error_types:
                relevant_errors = [
                    e for e in pkt.errors if any(t.lower() in e.lower() for t in error_types)
                ]

            for err in relevant_errors:
                sample_idx = int(pkt.timestamp * sample_rate)
                context_trace = _extract_context(original_trace, sample_idx, context_samples)

                error = {
                    "type": err,
                    "timestamp": pkt.timestamp,
                    "packet_index": i,
                    "address": None,
                    "data": pkt.data,
                    "context": f"Samples {sample_idx - context_samples} to {sample_idx + context_samples}",
                    "context_trace": context_trace,
                }
                errors.append(error)

    return packets, errors


def _decode_spi(
    trace: DigitalTrace,
    config: dict[str, Any],
    context_samples: int,
    error_types: list[str] | None,
    original_trace: WaveformTrace | DigitalTrace,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]]]:
    """Decode SPI protocol with error context.

    Args:
        trace: Digital trace to decode.
        config: SPI configuration.
        context_samples: Context window size.
        error_types: Error types to detect.
        original_trace: Original trace for context extraction.

    Returns:
        Tuple of (packets, errors).
    """
    from oscura.analyzers.protocols.spi import SPIDecoder

    cpol = config.get("clock_polarity", 0)
    cpha = config.get("clock_phase", 0)
    word_size = config.get("word_size", 8)
    bit_order = config.get("bit_order", "msb").lower()

    decoder = SPIDecoder(
        cpol=cpol,
        cpha=cpha,
        word_size=word_size,
        bit_order=bit_order,
    )

    # For single-channel decode, use trace data as both clock and data
    clk = trace.data
    mosi = trace.data

    packets = list(
        decoder.decode(
            clk=clk,
            mosi=mosi,
            sample_rate=trace.metadata.sample_rate,
        )
    )

    errors: list[dict[str, Any]] = []
    sample_rate = trace.metadata.sample_rate

    for i, pkt in enumerate(packets):
        if pkt.errors:
            relevant_errors = pkt.errors
            if error_types:
                relevant_errors = [
                    e for e in pkt.errors if any(t.lower() in e.lower() for t in error_types)
                ]

            for err in relevant_errors:
                sample_idx = int(pkt.timestamp * sample_rate)
                context_trace = _extract_context(original_trace, sample_idx, context_samples)

                error = {
                    "type": err,
                    "timestamp": pkt.timestamp,
                    "packet_index": i,
                    "mosi_data": pkt.data,
                    "context": f"Samples {sample_idx - context_samples} to {sample_idx + context_samples}",
                    "context_trace": context_trace,
                }
                errors.append(error)

    return packets, errors


def _decode_i2c(
    trace: DigitalTrace,
    config: dict[str, Any],
    context_samples: int,
    error_types: list[str] | None,
    original_trace: WaveformTrace | DigitalTrace,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]]]:
    """Decode I2C protocol with error context.

    Args:
        trace: Digital trace to decode.
        config: I2C configuration.
        context_samples: Context window size.
        error_types: Error types to detect.
        original_trace: Original trace for context extraction.

    Returns:
        Tuple of (packets, errors).
    """
    from oscura.analyzers.protocols.i2c import I2CDecoder

    address_format = config.get("address_format", "auto")

    # For single-channel, assume SDA and create synthetic SCL
    sda = trace.data
    sample_rate = trace.metadata.sample_rate

    edges = np.where(np.diff(sda.astype(int)) != 0)[0]

    if len(edges) < 20:
        return [], []

    # Create synthetic SCL
    scl = np.ones_like(sda, dtype=bool)
    for i, edge in enumerate(edges):
        if i % 2 == 0 and edge + 10 < len(scl):
            scl[edge : edge + 10] = False

    decoder = I2CDecoder(address_format=address_format)
    packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

    errors: list[dict[str, Any]] = []

    for i, pkt in enumerate(packets):
        if pkt.errors:
            relevant_errors = pkt.errors
            if error_types:
                relevant_errors = [
                    e for e in pkt.errors if any(t.lower() in e.lower() for t in error_types)
                ]

            for err in relevant_errors:
                sample_idx = int(pkt.timestamp * sample_rate)
                context_trace = _extract_context(original_trace, sample_idx, context_samples)

                addr = pkt.annotations.get("address", 0) if pkt.annotations else 0

                error = {
                    "type": err,
                    "timestamp": pkt.timestamp,
                    "packet_index": i,
                    "address": addr,
                    "data": pkt.data,
                    "context": f"Samples {sample_idx - context_samples} to {sample_idx + context_samples}",
                    "context_trace": context_trace,
                }
                errors.append(error)

    return packets, errors


def _decode_can(
    trace: DigitalTrace,
    config: dict[str, Any],
    context_samples: int,
    error_types: list[str] | None,
    original_trace: WaveformTrace | DigitalTrace,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]]]:
    """Decode CAN protocol with error context.

    Args:
        trace: Digital trace to decode.
        config: CAN configuration.
        context_samples: Context window size.
        error_types: Error types to detect.
        original_trace: Original trace for context extraction.

    Returns:
        Tuple of (packets, errors).
    """
    from oscura.analyzers.protocols.can import CANDecoder

    bitrate = config.get("baud_rate", 500000)
    sample_point = config.get("sample_point", 0.75)

    decoder = CANDecoder(bitrate=bitrate, sample_point=sample_point)
    packets = list(decoder.decode(trace))

    errors: list[dict[str, Any]] = []
    sample_rate = trace.metadata.sample_rate

    for i, pkt in enumerate(packets):
        if pkt.errors:
            relevant_errors = pkt.errors
            if error_types:
                relevant_errors = [
                    e for e in pkt.errors if any(t.lower() in e.lower() for t in error_types)
                ]

            for err in relevant_errors:
                sample_idx = int(pkt.timestamp * sample_rate)
                context_trace = _extract_context(original_trace, sample_idx, context_samples)

                arb_id = pkt.annotations.get("arbitration_id", 0) if pkt.annotations else 0

                error = {
                    "type": err,
                    "timestamp": pkt.timestamp,
                    "packet_index": i,
                    "arbitration_id": arb_id,
                    "data": pkt.data,
                    "context": f"Samples {sample_idx - context_samples} to {sample_idx + context_samples}",
                    "context_trace": context_trace,
                }
                errors.append(error)

    return packets, errors


__all__ = ["debug_protocol"]
