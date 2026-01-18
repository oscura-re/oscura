"""Oscura Decode Command implementing CLI-003.

Provides CLI for protocol decoding with automatic protocol detection and
error highlighting.


Example:
    $ oscura decode serial_capture.wfm
    $ oscura decode i2c_bus.wfm --protocol I2C
    $ oscura decode uart.wfm --protocol UART --baud-rate 115200
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np

from oscura.cli.main import format_output
from oscura.core.types import DigitalTrace, ProtocolPacket, WaveformTrace

logger = logging.getLogger("oscura.cli.decode")


@click.command()  # type: ignore[misc]
@click.argument("file", type=click.Path(exists=True))  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--protocol",
    type=click.Choice(["uart", "spi", "i2c", "can", "auto"], case_sensitive=False),
    default="auto",
    help="Protocol type (default: auto-detect).",
)
@click.option(  # type: ignore[misc]
    "--baud-rate",
    type=int,
    default=None,
    help="Baud rate for UART (auto-detect if not specified).",
)
@click.option(  # type: ignore[misc]
    "--parity",
    type=click.Choice(["none", "even", "odd"], case_sensitive=False),
    default="none",
    help="Parity for UART (default: none).",
)
@click.option(  # type: ignore[misc]
    "--stop-bits",
    type=click.Choice(["1", "2"]),
    default="1",
    help="Stop bits for UART (default: 1).",
)
@click.option(  # type: ignore[misc]
    "--show-errors",
    is_flag=True,
    help="Show only errors with context.",
)
@click.option(  # type: ignore[misc]
    "--output",
    type=click.Choice(["json", "csv", "html", "table"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.pass_context  # type: ignore[misc]
def decode(
    ctx: click.Context,
    file: str,
    protocol: str,
    baud_rate: int | None,
    parity: str,
    stop_bits: str,
    show_errors: bool,
    output: str,
) -> None:
    """Decode serial protocol data.

    Automatically detects and decodes common serial protocols (UART, SPI, I2C, CAN).
    Can highlight errors with surrounding context for debugging.

    Args:
        ctx: Click context object.
        file: Path to waveform file to decode.
        protocol: Protocol type (uart, spi, i2c, can, auto).
        baud_rate: Baud rate for UART (None for auto-detect).
        parity: Parity setting for UART (none, even, odd).
        stop_bits: Number of stop bits for UART (1 or 2).
        show_errors: Show only packets with errors.
        output: Output format (json, csv, html, table).

    Raises:
        Exception: If decoding fails or file cannot be loaded.

    Examples:

        \b
        # Auto-detect and decode protocol
        $ oscura decode serial_capture.wfm

        \b
        # Decode specific protocol with parameters
        $ oscura decode uart.wfm \\
            --protocol UART \\
            --baud-rate 9600 \\
            --parity even \\
            --stop-bits 2

        \b
        # Show only errors for debugging
        $ oscura decode problematic.wfm --show-errors

        \b
        # Generate JSON output
        $ oscura decode i2c.wfm --protocol I2C --output json
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Decoding: {file}")
        logger.info(f"Protocol: {protocol}")
        if protocol.lower() == "uart" and baud_rate:
            logger.info(f"Baud rate: {baud_rate}")

    try:
        # Import here to avoid circular imports
        from oscura.loaders import load

        # Load the trace
        logger.debug(f"Loading trace from {file}")
        trace = load(file)

        # Perform protocol decoding
        results = _perform_decoding(
            trace=trace,  # type: ignore[arg-type]
            protocol=protocol,
            baud_rate=baud_rate,
            parity=parity,
            stop_bits=int(stop_bits),
            show_errors=show_errors,
        )

        # Add metadata
        results["file"] = str(Path(file).name)

        # Output results
        formatted = format_output(results, output)
        click.echo(formatted)

    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _to_digital(trace: WaveformTrace | DigitalTrace) -> DigitalTrace:
    """Convert waveform trace to digital trace.

    Args:
        trace: Input trace (waveform or digital).

    Returns:
        Digital trace with boolean data.
    """
    if isinstance(trace, DigitalTrace):
        return trace

    # Use midpoint threshold for digitization
    data = trace.data
    threshold = (np.max(data) + np.min(data)) / 2
    digital_data = data > threshold

    return DigitalTrace(
        data=digital_data,
        metadata=trace.metadata,
    )


def _perform_decoding(
    trace: WaveformTrace | DigitalTrace,
    protocol: str,
    baud_rate: int | None,
    parity: str,
    stop_bits: int,
    show_errors: bool,
) -> dict[str, Any]:
    """Perform protocol decoding using actual decoders.

    Args:
        trace: Trace to decode.
        protocol: Protocol type or 'auto'.
        baud_rate: Optional baud rate for UART.
        parity: Parity setting for UART.
        stop_bits: Stop bits for UART.
        show_errors: Whether to show only errors.

    Returns:
        Dictionary of decoding results.
    """
    # Import protocol decoders
    from oscura.inference.protocol import detect_protocol

    sample_rate = trace.metadata.sample_rate
    duration_ms = len(trace.data) / sample_rate * 1e3

    results: dict[str, Any] = {
        "sample_rate": f"{sample_rate / 1e6:.1f} MHz",
        "samples": len(trace.data),
        "duration": f"{duration_ms:.3f} ms",
    }

    # Auto-detect protocol if requested
    detected_protocol = protocol
    detection_confidence = 1.0

    if protocol.lower() == "auto":
        try:
            detection = detect_protocol(trace, min_confidence=0.5, return_candidates=True)  # type: ignore[arg-type]
            detected_protocol = detection["protocol"].lower()
            detection_confidence = detection["confidence"]
            results["auto_detection"] = {
                "protocol": detection["protocol"],
                "confidence": f"{detection_confidence:.1%}",
                "candidates": [
                    {"protocol": c["protocol"], "confidence": f"{c['confidence']:.1%}"}
                    for c in detection.get("candidates", [])[:3]
                ],
            }
            # Extract config suggestions
            if "config" in detection:
                if detected_protocol == "uart" and baud_rate is None:
                    baud_rate = detection["config"].get("baud_rate")
        except Exception as e:
            logger.warning(f"Auto-detection failed: {e}, defaulting to UART")
            detected_protocol = "uart"
            detection_confidence = 0.0

    results["protocol"] = detected_protocol.upper()

    # Convert to digital trace for decoding
    digital_trace = _to_digital(trace)

    # Decode based on protocol
    packets: list[ProtocolPacket] = []
    errors: list[dict[str, Any]] = []

    if detected_protocol == "uart":
        packets, errors, protocol_info = _decode_uart(
            digital_trace, baud_rate, parity, stop_bits, show_errors
        )
        results.update(protocol_info)

    elif detected_protocol == "spi":
        packets, errors, protocol_info = _decode_spi(digital_trace, show_errors)
        results.update(protocol_info)

    elif detected_protocol == "i2c":
        packets, errors, protocol_info = _decode_i2c(digital_trace, show_errors)
        results.update(protocol_info)

    elif detected_protocol == "can":
        packets, errors, protocol_info = _decode_can(digital_trace, baud_rate, show_errors)
        results.update(protocol_info)

    # Filter to errors only if requested
    if show_errors:
        packets = [p for p in packets if p.errors]

    # Summarize results
    results["packets_decoded"] = len(packets)
    results["errors_found"] = len(errors)

    # Add packet details
    results["packets"] = [
        {
            "index": i,
            "timestamp": f"{p.timestamp * 1e3:.3f} ms",
            "data": p.data.hex() if p.data else "",
            "errors": p.errors,
            **{k: v for k, v in (p.annotations or {}).items() if k != "data_bits"},
        }
        for i, p in enumerate(packets[:100])  # Limit to first 100 packets
    ]

    if len(packets) > 100:
        results["note"] = f"Showing first 100 of {len(packets)} packets"

    # Add error details if any
    if errors:
        results["error_details"] = errors[:20]  # Limit to first 20 errors

    return results


def _decode_uart(
    trace: DigitalTrace,
    baud_rate: int | None,
    parity: str,
    stop_bits: int,
    show_errors: bool,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]], dict[str, Any]]:
    """Decode UART protocol.

    Args:
        trace: Digital trace to decode.
        baud_rate: Baud rate (None for auto-detect).
        parity: Parity mode.
        stop_bits: Number of stop bits.
        show_errors: Whether to filter to errors only.

    Returns:
        Tuple of (packets, errors, protocol_info).
    """
    from oscura.analyzers.protocols.uart import UARTDecoder

    # Create decoder with parameters
    decoder = UARTDecoder(
        baudrate=baud_rate or 0,  # 0 triggers auto-detection
        data_bits=8,
        parity=parity,  # type: ignore[arg-type]
        stop_bits=stop_bits,
    )

    packets = list(decoder.decode(trace))
    errors = []

    # Extract errors from packets
    for i, pkt in enumerate(packets):
        if pkt.errors:
            for err in pkt.errors:
                errors.append(
                    {
                        "packet_index": i,
                        "timestamp": f"{pkt.timestamp * 1e3:.3f} ms",
                        "type": err,
                        "data": pkt.data.hex() if pkt.data else "",
                    }
                )

    # Determine actual baud rate used
    actual_baud = decoder._baudrate if hasattr(decoder, "_baudrate") else baud_rate

    protocol_info = {
        "baud_rate": actual_baud,
        "parity": parity,
        "stop_bits": stop_bits,
        "data_bits": 8,
    }

    return packets, errors, protocol_info


def _decode_spi(
    trace: DigitalTrace,
    show_errors: bool,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]], dict[str, Any]]:
    """Decode SPI protocol.

    Note: SPI requires multiple signals (CLK, MOSI, optionally MISO, CS).
    This function assumes the trace contains clock data and will attempt
    to decode what's available.

    Args:
        trace: Digital trace to decode (assumed to be CLK or combined).
        show_errors: Whether to filter to errors only.

    Returns:
        Tuple of (packets, errors, protocol_info).
    """
    from oscura.analyzers.protocols.spi import SPIDecoder

    # For single-channel decode, we can only analyze timing
    # Full SPI decode requires separate CLK, MOSI, MISO channels
    decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

    # Create MOSI from the trace data (treating as data line)
    clk = trace.data
    mosi = trace.data  # Same data for single-channel analysis

    packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=trace.metadata.sample_rate))
    errors = []

    for i, pkt in enumerate(packets):
        if pkt.errors:
            for err in pkt.errors:
                errors.append(
                    {
                        "packet_index": i,
                        "timestamp": f"{pkt.timestamp * 1e3:.3f} ms",
                        "type": err,
                    }
                )

    # Estimate clock frequency from edge timing
    edges = np.where(np.diff(clk.astype(int)) != 0)[0]
    if len(edges) > 1:
        avg_period = np.mean(np.diff(edges)) / trace.metadata.sample_rate
        clock_freq = 1 / (2 * avg_period) if avg_period > 0 else 0
    else:
        clock_freq = 0

    protocol_info = {
        "clock_frequency": f"{clock_freq / 1e6:.2f} MHz" if clock_freq > 0 else "Unknown",
        "mode": "0 (CPOL=0, CPHA=0)",
        "word_size": 8,
        "note": "Single-channel decode. For full SPI decode, provide separate CLK/MOSI/MISO signals.",
    }

    return packets, errors, protocol_info


def _decode_i2c(
    trace: DigitalTrace,
    show_errors: bool,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]], dict[str, Any]]:
    """Decode I2C protocol.

    Note: I2C requires two signals (SCL, SDA). This function assumes
    the trace is SDA and attempts to detect SCL from timing patterns.

    Args:
        trace: Digital trace to decode.
        show_errors: Whether to filter to errors only.

    Returns:
        Tuple of (packets, errors, protocol_info).
    """
    from oscura.analyzers.protocols.i2c import I2CDecoder

    # For single-channel, assume it's SDA and create synthetic SCL from edges
    sda = trace.data
    sample_rate = trace.metadata.sample_rate

    # Try to find clock pattern from edge timing
    edges = np.where(np.diff(sda.astype(int)) != 0)[0]
    if len(edges) < 20:
        # Not enough edges for I2C
        return [], [], {"error": "Insufficient edges for I2C decode"}

    # Create synthetic SCL (toggle at each edge for analysis)
    scl = np.ones_like(sda, dtype=bool)
    for i, edge in enumerate(edges):
        if i % 2 == 0 and edge + 1 < len(scl):
            scl[edge : edge + 10] = False  # Create clock pulses

    decoder = I2CDecoder()
    packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))
    errors = []

    addresses_seen: set[int] = set()
    for i, pkt in enumerate(packets):
        addr = pkt.annotations.get("address", 0) if pkt.annotations else 0
        addresses_seen.add(addr)

        if pkt.errors:
            for err in pkt.errors:
                errors.append(
                    {
                        "packet_index": i,
                        "timestamp": f"{pkt.timestamp * 1e3:.3f} ms",
                        "type": err,
                        "address": f"0x{addr:02X}",
                    }
                )

    # Estimate clock rate from edge intervals
    if len(edges) > 1:
        avg_interval = np.mean(np.diff(edges)) / sample_rate
        clock_rate = 1 / (2 * avg_interval) if avg_interval > 0 else 0
    else:
        clock_rate = 0

    protocol_info = {
        "clock_frequency": f"{clock_rate / 1e3:.1f} kHz" if clock_rate > 0 else "Unknown",
        "addresses_seen": [f"0x{a:02X}" for a in sorted(addresses_seen)],
        "transactions": len(packets),
        "note": "Single-channel decode. For accurate I2C decode, provide separate SCL/SDA signals.",
    }

    return packets, errors, protocol_info


def _decode_can(
    trace: DigitalTrace,
    baud_rate: int | None,
    show_errors: bool,
) -> tuple[list[ProtocolPacket], list[dict[str, Any]], dict[str, Any]]:
    """Decode CAN protocol.

    Args:
        trace: Digital trace to decode.
        baud_rate: CAN bit rate (None for common rate detection).
        show_errors: Whether to filter to errors only.

    Returns:
        Tuple of (packets, errors, protocol_info).
    """
    from oscura.analyzers.protocols.can import CANDecoder

    # Try common CAN baud rates if not specified
    if baud_rate is None:
        common_rates = [500000, 250000, 125000, 1000000]
        best_rate = 500000
        max_packets = 0

        for rate in common_rates:
            try:
                decoder = CANDecoder(bitrate=rate)
                test_packets = list(decoder.decode(trace))
                if len(test_packets) > max_packets:
                    max_packets = len(test_packets)
                    best_rate = rate
            except Exception:
                continue

        baud_rate = best_rate

    decoder = CANDecoder(bitrate=baud_rate)
    packets = list(decoder.decode(trace))
    errors = []

    arbitration_ids: set[int] = set()
    for i, pkt in enumerate(packets):
        arb_id = pkt.annotations.get("arbitration_id", 0) if pkt.annotations else 0
        arbitration_ids.add(arb_id)

        if pkt.errors:
            for err in pkt.errors:
                errors.append(
                    {
                        "packet_index": i,
                        "timestamp": f"{pkt.timestamp * 1e3:.3f} ms",
                        "type": err,
                        "arbitration_id": f"0x{arb_id:03X}",
                    }
                )

    protocol_info = {
        "bit_rate": f"{baud_rate / 1000:.0f} kbps",
        "messages": len(packets),
        "arbitration_ids": [f"0x{a:03X}" for a in sorted(arbitration_ids)[:10]],
        "extended_frames": sum(
            1 for p in packets if p.annotations and p.annotations.get("is_extended")
        ),
    }

    if len(arbitration_ids) > 10:
        protocol_info["note"] = f"Showing first 10 of {len(arbitration_ids)} arbitration IDs"

    return packets, errors, protocol_info
