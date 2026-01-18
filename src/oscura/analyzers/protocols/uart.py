"""UART protocol decoder.

This module provides UART/RS-232 protocol decoding with auto-baud
detection and configurable parameters.


Example:
    >>> from oscura.analyzers.protocols.uart import UARTDecoder
    >>> decoder = UARTDecoder(baudrate=115200)
    >>> for packet in decoder.decode(trace):
    ...     print(f"Data: {packet.data.hex()}")

References:
    EIA/TIA-232-F Standard
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    OptionDef,
)
from oscura.core.types import (
    DigitalTrace,
    ProtocolPacket,
    TraceMetadata,
    WaveformTrace,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class UARTDecoder(AsyncDecoder):
    """UART protocol decoder.

    Decodes UART data with configurable parameters including
    auto-baud detection, data bits, parity, and stop bits.

    Attributes:
        id: "uart"
        name: "UART"
        channels: [rx] (required), [tx] (optional)

    Example:
        >>> decoder = UARTDecoder(baudrate=115200, data_bits=8, parity="none")
        >>> for packet in decoder.decode(trace):
        ...     print(f"Byte: 0x{packet.data[0]:02X}")
    """

    id = "uart"
    name = "UART"
    longname = "Universal Asynchronous Receiver/Transmitter"
    desc = "UART/RS-232 serial protocol decoder"

    channels = [  # noqa: RUF012
        ChannelDef("rx", "RX", "Receive data line", required=True),
    ]

    optional_channels = [  # noqa: RUF012
        ChannelDef("tx", "TX", "Transmit data line", required=False),
    ]

    options = [  # noqa: RUF012
        OptionDef("baudrate", "Baud rate", "Bits per second", default=0, values=None),
        OptionDef(
            "data_bits",
            "Data bits",
            "Number of data bits",
            default=8,
            values=[5, 6, 7, 8, 9],
        ),
        OptionDef(
            "parity",
            "Parity",
            "Parity mode",
            default="none",
            values=["none", "odd", "even", "mark", "space"],
        ),
        OptionDef(
            "stop_bits",
            "Stop bits",
            "Number of stop bits",
            default=1,
            values=[1, 1.5, 2],
        ),
        OptionDef(
            "bit_order",
            "Bit order",
            "Data bit order",
            default="lsb",
            values=["lsb", "msb"],
        ),
        OptionDef("idle_level", "Idle level", "Idle line level", default=1, values=[0, 1]),
    ]

    annotations = [  # noqa: RUF012
        ("bit", "Bit value"),
        ("start", "Start bit"),
        ("data", "Data bits"),
        ("parity", "Parity bit"),
        ("stop", "Stop bit"),
        ("byte", "Decoded byte"),
        ("error", "Error"),
    ]

    def __init__(
        self,
        baudrate: int = 0,
        data_bits: int = 8,
        parity: Literal["none", "odd", "even", "mark", "space"] = "none",
        stop_bits: float = 1,
        bit_order: Literal["lsb", "msb"] = "lsb",
        idle_level: int = 1,
    ) -> None:
        """Initialize UART decoder.

        Args:
            baudrate: Baud rate in bps. 0 for auto-detect.
            data_bits: Number of data bits (5-9).
            parity: Parity mode.
            stop_bits: Number of stop bits (1, 1.5, 2).
            bit_order: Bit order ("lsb" or "msb").
            idle_level: Idle line level (0 or 1).
        """
        super().__init__(
            baudrate=baudrate,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            bit_order=bit_order,
            idle_level=idle_level,
        )
        self._data_bits = data_bits
        self._parity = parity
        self._stop_bits = stop_bits
        self._bit_order = bit_order
        self._idle_level = idle_level

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode UART data from trace.

        Args:
            trace: Input digital trace.
            **channels: Additional channel data.

        Yields:
            Decoded UART bytes as ProtocolPacket objects.

        Example:
            >>> decoder = UARTDecoder(baudrate=9600)
            >>> for packet in decoder.decode(trace):
            ...     print(f"Byte: {packet.data.hex()}")
        """
        # Convert to digital if needed
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            digital_trace = to_digital(trace, threshold="auto")
        else:
            digital_trace = trace

        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        # Auto-detect baud rate if not specified
        if self._baudrate == 0:
            from oscura.utils.autodetect import detect_baud_rate

            self._baudrate = detect_baud_rate(digital_trace)  # type: ignore[assignment]
            if self._baudrate == 0:
                self._baudrate = 9600  # Fallback

        bit_period = sample_rate / self._baudrate
        half_bit = bit_period / 2

        # Frame structure
        frame_bits = 1 + self._data_bits  # Start + data
        if self._parity != "none":
            frame_bits += 1
        frame_bits += self._stop_bits  # type: ignore[assignment]

        idx = 0
        frame_num = 0

        while idx < len(data) - int(frame_bits * bit_period):
            # Look for start bit (transition from idle)
            start_idx = self._find_start_bit(data, idx)
            if start_idx is None:
                break

            # Sample at center of each bit
            sample_points = []
            for bit_num in range(int(frame_bits)):
                sample_idx = int(start_idx + half_bit + bit_num * bit_period)
                if sample_idx < len(data):
                    sample_points.append(sample_idx)

            if len(sample_points) < 1 + self._data_bits:
                break

            # Verify start bit (should be opposite of idle)
            start_bit = data[sample_points[0]]
            if (self._idle_level == 1 and start_bit) or (self._idle_level == 0 and not start_bit):
                # Not a valid start bit
                idx = start_idx + 1
                continue

            # Extract data bits
            data_value = 0
            data_bits = []

            for i in range(self._data_bits):
                bit_idx = sample_points[1 + i]
                bit_val = 1 if data[bit_idx] else 0
                data_bits.append(bit_val)

                if self._bit_order == "lsb":
                    data_value |= bit_val << i
                else:
                    data_value |= bit_val << (self._data_bits - 1 - i)

            # Check parity if enabled
            errors = []
            parity_idx = 1 + self._data_bits

            if self._parity != "none" and parity_idx < len(sample_points):
                parity_bit = 1 if data[sample_points[parity_idx]] else 0
                ones_count = sum(data_bits)

                if self._parity == "odd":
                    expected = (ones_count + 1) % 2
                elif self._parity == "even":
                    expected = ones_count % 2
                elif self._parity == "mark":
                    expected = 1
                else:  # space
                    expected = 0

                if parity_bit != expected:
                    errors.append("Parity error")

            # Verify stop bit(s)
            stop_idx = parity_idx + (1 if self._parity != "none" else 0)
            if stop_idx < len(sample_points):
                stop_bit = data[sample_points[stop_idx]]
                expected_stop = self._idle_level == 1

                if stop_bit != expected_stop:
                    errors.append("Framing error")

            # Calculate timestamps
            start_time = start_idx / sample_rate
            end_time = (start_idx + frame_bits * bit_period) / sample_rate

            # Add annotations
            self.put_annotation(
                start_time,
                start_time + bit_period / sample_rate,
                AnnotationLevel.BITS,
                "START",
            )

            for i, bit_val in enumerate(data_bits):
                bit_start = start_time + (1 + i) * bit_period / sample_rate
                bit_end = bit_start + bit_period / sample_rate
                self.put_annotation(
                    bit_start,
                    bit_end,
                    AnnotationLevel.BITS,
                    str(bit_val),
                )

            self.put_annotation(
                start_time,
                end_time,
                AnnotationLevel.BYTES,
                f"0x{data_value:02X}",
                data=bytes([data_value]),
            )

            # Create packet
            packet = ProtocolPacket(
                timestamp=start_time,
                protocol="uart",
                data=bytes([data_value]),
                annotations={
                    "frame_num": frame_num,
                    "data_bits": data_bits,
                    "baudrate": self._baudrate,
                },
                errors=errors,
            )

            self.put_packet(start_time, bytes([data_value]), packet.annotations, errors)

            yield packet

            frame_num += 1
            # Advance to the end of the frame
            # Use the last sample point + 1 to avoid re-detecting the same frame
            last_sample = sample_points[-1] if sample_points else start_idx
            idx = last_sample + 1

    def _find_start_bit(
        self,
        data: NDArray[np.bool_],
        start_idx: int,
    ) -> int | None:
        """Find start bit transition.

        Args:
            data: Digital data array.
            start_idx: Index to start searching.

        Returns:
            Index of start bit, or None if not found.
        """
        search = data[start_idx:]

        if self._idle_level == 1:
            # Look for falling edge (high to low)
            transitions = np.where(search[:-1] & ~search[1:])[0]
        else:
            # Look for rising edge (low to high)
            transitions = np.where(~search[:-1] & search[1:])[0]

        if len(transitions) == 0:
            return None

        # Return index of first sample after the transition (start of start bit)
        # transitions[0] is the last idle-level sample before the edge
        return int(start_idx + transitions[0] + 1)


def decode_uart(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    baudrate: int | None = None,
    data_bits: Literal[5, 6, 7, 8, 9] = 8,
    parity: Literal["none", "odd", "even", "mark", "space"] = "none",
    stop_bits: Literal[1, 1.5, 2] = 1,  # type: ignore[valid-type]
    idle_level: Literal[0, 1] = 1,
) -> list[ProtocolPacket]:
    """Convenience function to decode UART data.

    Args:
        data: UART signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        baudrate: Baud rate (None for auto-detection).
        data_bits: Number of data bits per frame.
        parity: Parity mode.
        stop_bits: Number of stop bits.
        idle_level: Idle line level.

    Returns:
        List of decoded UART bytes.

    Example:
        >>> packets = decode_uart(signal, sample_rate=10e6, baudrate=115200)
        >>> for pkt in packets:
        ...     print(f"Byte: 0x{pkt.data[0]:02X}")
    """
    decoder = UARTDecoder(
        baudrate=baudrate if baudrate is not None else 0,  # 0 for auto-detect
        data_bits=data_bits,
        parity=parity,
        stop_bits=stop_bits,
        idle_level=idle_level,
    )
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        trace = DigitalTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        return list(decoder.decode(trace))


__all__ = ["UARTDecoder", "decode_uart"]
