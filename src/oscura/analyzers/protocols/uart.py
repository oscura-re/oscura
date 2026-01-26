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

    channels = [
        ChannelDef("rx", "RX", "Receive data line", required=True),
    ]

    optional_channels = [
        ChannelDef("tx", "TX", "Transmit data line", required=False),
    ]

    options = [
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

    annotations = [
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
        digital_trace = self._convert_to_digital(trace)
        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        self._auto_detect_baudrate(digital_trace)
        bit_period = sample_rate / self._baudrate
        frame_bits = self._calculate_frame_bits()

        idx = 0
        frame_num = 0

        while idx < len(data) - int(frame_bits * bit_period):
            start_idx = self._find_start_bit(data, idx)
            if start_idx is None:
                break

            sample_points = self._get_sample_points(start_idx, bit_period, frame_bits, len(data))
            if len(sample_points) < 1 + self._data_bits:
                break

            if not self._verify_start_bit(data, sample_points):
                idx = start_idx + 1
                continue

            data_value, data_bits = self._extract_data_bits(data, sample_points)
            errors = self._validate_frame(data, sample_points, data_bits)

            packet = self._build_packet(
                start_idx,
                sample_rate,
                bit_period,
                frame_bits,
                data_value,
                data_bits,
                frame_num,
                errors,
            )

            yield packet

            frame_num += 1
            last_sample = sample_points[-1] if sample_points else start_idx
            idx = last_sample + 1

    def _convert_to_digital(self, trace: DigitalTrace | WaveformTrace) -> DigitalTrace:
        """Convert trace to digital format if needed.

        Args:
            trace: Input trace.

        Returns:
            Digital trace.
        """
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            return to_digital(trace, threshold="auto")
        return trace

    def _auto_detect_baudrate(self, trace: DigitalTrace) -> None:
        """Auto-detect baud rate if not specified.

        Args:
            trace: Digital trace for detection.
        """
        if self._baudrate == 0:
            from oscura.utils.autodetect import detect_baud_rate

            self._baudrate = detect_baud_rate(trace)  # type: ignore[assignment]
            if self._baudrate == 0:
                self._baudrate = 9600  # Fallback

    def _calculate_frame_bits(self) -> float:
        """Calculate total bits per frame.

        Returns:
            Number of bits in frame.
        """
        frame_bits = 1 + self._data_bits  # Start + data
        if self._parity != "none":
            frame_bits += 1
        frame_bits += self._stop_bits  # type: ignore[assignment]
        return frame_bits

    def _get_sample_points(
        self, start_idx: int, bit_period: float, frame_bits: float, data_len: int
    ) -> list[int]:
        """Get sample points for each bit in frame.

        Args:
            start_idx: Frame start index.
            bit_period: Samples per bit.
            frame_bits: Total bits in frame.
            data_len: Length of data array.

        Returns:
            List of sample indices.
        """
        half_bit = bit_period / 2
        sample_points = []
        for bit_num in range(int(frame_bits)):
            sample_idx = int(start_idx + half_bit + bit_num * bit_period)
            if sample_idx < data_len:
                sample_points.append(sample_idx)
        return sample_points

    def _verify_start_bit(self, data: NDArray[np.bool_], sample_points: list[int]) -> bool:
        """Verify start bit is valid.

        Args:
            data: Digital data array.
            sample_points: Sample point indices.

        Returns:
            True if start bit valid.
        """
        start_bit = data[sample_points[0]]
        expected_start = self._idle_level == 0
        return bool(start_bit == expected_start)

    def _extract_data_bits(
        self, data: NDArray[np.bool_], sample_points: list[int]
    ) -> tuple[int, list[int]]:
        """Extract data bits and convert to byte value.

        Args:
            data: Digital data array.
            sample_points: Sample point indices.

        Returns:
            Tuple of (byte_value, bit_list).
        """
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

        return data_value, data_bits

    def _validate_frame(
        self, data: NDArray[np.bool_], sample_points: list[int], data_bits: list[int]
    ) -> list[str]:
        """Validate parity and stop bits.

        Args:
            data: Digital data array.
            sample_points: Sample point indices.
            data_bits: Extracted data bits.

        Returns:
            List of error messages.
        """
        errors = []
        parity_idx = 1 + self._data_bits

        # Check parity
        if self._parity != "none" and parity_idx < len(sample_points):
            parity_bit = 1 if data[sample_points[parity_idx]] else 0
            expected_parity = self._calculate_parity(data_bits)
            if parity_bit != expected_parity:
                errors.append("Parity error")

        # Verify stop bit
        stop_idx = parity_idx + (1 if self._parity != "none" else 0)
        if stop_idx < len(sample_points):
            stop_bit = data[sample_points[stop_idx]]
            expected_stop = self._idle_level == 1
            if stop_bit != expected_stop:
                errors.append("Framing error")

        return errors

    def _calculate_parity(self, data_bits: list[int]) -> int:
        """Calculate expected parity bit.

        Args:
            data_bits: Data bit values.

        Returns:
            Expected parity bit (0 or 1).
        """
        ones_count = sum(data_bits)

        if self._parity == "odd":
            return (ones_count + 1) % 2
        elif self._parity == "even":
            return ones_count % 2
        elif self._parity == "mark":
            return 1
        else:  # space
            return 0

    def _build_packet(
        self,
        start_idx: int,
        sample_rate: float,
        bit_period: float,
        frame_bits: float,
        data_value: int,
        data_bits: list[int],
        frame_num: int,
        errors: list[str],
    ) -> ProtocolPacket:
        """Build protocol packet with annotations.

        Args:
            start_idx: Frame start index.
            sample_rate: Sample rate in Hz.
            bit_period: Samples per bit.
            frame_bits: Total bits in frame.
            data_value: Decoded byte value.
            data_bits: Individual bit values.
            frame_num: Frame number.
            errors: List of errors.

        Returns:
            Protocol packet.
        """
        start_time = start_idx / sample_rate

        # Add annotations
        self._add_frame_annotations(start_time, bit_period, sample_rate, data_value, data_bits)

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
        return packet

    def _add_frame_annotations(
        self,
        start_time: float,
        bit_period: float,
        sample_rate: float,
        data_value: int,
        data_bits: list[int],
    ) -> None:
        """Add bit-level and byte-level annotations.

        Args:
            start_time: Frame start time.
            bit_period: Samples per bit.
            sample_rate: Sample rate in Hz.
            data_value: Decoded byte value.
            data_bits: Individual bit values.
        """
        # Start bit annotation
        self.put_annotation(
            start_time,
            start_time + bit_period / sample_rate,
            AnnotationLevel.BITS,
            "START",
        )

        # Data bit annotations
        for i, bit_val in enumerate(data_bits):
            bit_start = start_time + (1 + i) * bit_period / sample_rate
            bit_end = bit_start + bit_period / sample_rate
            self.put_annotation(
                bit_start,
                bit_end,
                AnnotationLevel.BITS,
                str(bit_val),
            )

        # Byte annotation
        end_time = start_time + (1 + len(data_bits)) * bit_period / sample_rate
        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.BYTES,
            f"0x{data_value:02X}",
            data=bytes([data_value]),
        )

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
