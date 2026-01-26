"""LIN protocol decoder.

This module provides Local Interconnect Network (LIN) automotive protocol
decoding for LIN 1.x and 2.x frames.


Example:
    >>> from oscura.analyzers.protocols.lin import LINDecoder
    >>> decoder = LINDecoder(baudrate=19200)
    >>> for packet in decoder.decode(trace):
    ...     print(f"ID: 0x{packet.annotations['frame_id']:02X}")

References:
    LIN Specification 1.3, 2.0, 2.1, 2.2A
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    OptionDef,
)
from oscura.core.types import DigitalTrace, ProtocolPacket, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray


class LINVersion(Enum):
    """LIN protocol version."""

    LIN_1X = "1.x"
    LIN_2X = "2.x"


class LINDecoder(AsyncDecoder):
    """LIN protocol decoder.

    Decodes LIN bus frames with sync field validation, identifier extraction,
    and checksum validation for both LIN 1.x (classic) and 2.x (enhanced).

    Attributes:
        id: "lin"
        name: "LIN"
        channels: [bus] (required)

    Example:
        >>> decoder = LINDecoder(baudrate=19200, version="2.x")
        >>> for packet in decoder.decode(trace):
        ...     print(f"ID: 0x{packet.annotations['frame_id']:02X}")
    """

    id = "lin"
    name = "LIN"
    longname = "Local Interconnect Network"
    desc = "LIN automotive bus protocol decoder"

    channels = [
        ChannelDef("bus", "BUS", "LIN bus signal", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef(
            "baudrate",
            "Baud rate",
            "Bits per second",
            default=19200,
            values=[9600, 19200, 20000],
        ),
        OptionDef(
            "version",
            "LIN version",
            "Protocol version",
            default="2.x",
            values=["1.x", "2.x"],
        ),
    ]

    annotations = [
        ("sync", "Sync field"),
        ("pid", "Protected identifier"),
        ("data", "Data bytes"),
        ("checksum", "Checksum"),
        ("error", "Error"),
    ]

    def __init__(
        self,
        baudrate: int = 19200,
        version: Literal["1.x", "2.x"] = "2.x",
    ) -> None:
        """Initialize LIN decoder.

        Args:
            baudrate: Baud rate in bps (9600, 19200, 20000).
            version: LIN version ("1.x" or "2.x").
        """
        super().__init__(baudrate=baudrate, version=version)
        self._version = LINVersion.LIN_1X if version == "1.x" else LINVersion.LIN_2X

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode LIN frames from trace.

        Args:
            trace: Input digital trace.
            **channels: Additional channel data.

        Yields:
            Decoded LIN frames as ProtocolPacket objects.

        Example:
            >>> decoder = LINDecoder(baudrate=19200)
            >>> for packet in decoder.decode(trace):
            ...     print(f"Data: {packet.data.hex()}")
        """
        digital_trace = self._prepare_digital_trace(trace)
        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        bit_period = sample_rate / self._baudrate
        half_bit = bit_period / 2

        idx = 0
        frame_num = 0

        while idx < len(data):
            frame_result = self._try_decode_frame(
                data, idx, bit_period, half_bit, sample_rate, frame_num
            )

            if frame_result is None:
                break

            packet, next_idx = frame_result
            yield packet

            frame_num += 1
            idx = next_idx

    def _prepare_digital_trace(self, trace: DigitalTrace | WaveformTrace) -> DigitalTrace:
        """Convert trace to digital format if needed."""
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            return to_digital(trace, threshold="auto")
        return trace

    def _try_decode_frame(
        self,
        data: NDArray[np.bool_],
        idx: int,
        bit_period: float,
        half_bit: float,
        sample_rate: float,
        frame_num: int,
    ) -> tuple[ProtocolPacket, int] | None:
        """Attempt to decode a single LIN frame."""
        break_start = self._find_break_field(data, idx, bit_period)
        if break_start is None:
            return None

        sync_start_idx = self._find_sync_start(data, break_start)
        if sync_start_idx >= len(data):
            return None

        sync_byte, sync_errors = self._decode_sync_field(data, sync_start_idx, bit_period, half_bit)
        pid_byte, pid_errors, frame_id = self._decode_pid_field(
            data, sync_start_idx, bit_period, half_bit
        )

        if pid_byte is None:
            return None

        data_length = self._get_data_length(frame_id)
        data_bytes, data_errors = self._decode_data_fields(
            data, sync_start_idx, bit_period, half_bit, data_length
        )
        checksum_byte, checksum_errors = self._decode_checksum_field(
            data, sync_start_idx, bit_period, half_bit, data_length, frame_id, data_bytes
        )

        packet = self._create_lin_packet(
            break_start,
            sync_start_idx,
            bit_period,
            data_length,
            sample_rate,
            sync_errors,
            pid_errors,
            data_errors,
            checksum_errors,
            frame_num,
            frame_id,
            pid_byte,
            checksum_byte,
            data_bytes,
        )

        next_idx = int(sync_start_idx + (10 + 10 + data_length * 10 + 10) * bit_period)
        return packet, next_idx

    def _find_sync_start(self, data: NDArray[np.bool_], break_start: int) -> int:
        """Find start of sync byte after break field."""
        sync_start_idx = break_start
        while sync_start_idx < len(data) and not data[sync_start_idx]:
            sync_start_idx += 1

        while sync_start_idx < len(data) and data[sync_start_idx]:
            sync_start_idx += 1

        return sync_start_idx

    def _decode_sync_field(
        self,
        data: NDArray[np.bool_],
        sync_start_idx: int,
        bit_period: float,
        half_bit: float,
    ) -> tuple[int, list[str]]:
        """Decode and validate sync field."""
        sync_byte, sync_errors = self._decode_byte(data, sync_start_idx, bit_period, half_bit)

        if sync_byte != 0x55:
            sync_errors.append(f"Invalid sync field: 0x{sync_byte:02X} (expected 0x55)")

        return sync_byte, sync_errors

    def _decode_pid_field(
        self,
        data: NDArray[np.bool_],
        sync_start_idx: int,
        bit_period: float,
        half_bit: float,
    ) -> tuple[int | None, list[str], int]:
        """Decode and validate PID field."""
        pid_start_idx = int(sync_start_idx + 10 * bit_period)
        if pid_start_idx >= len(data):
            return None, [], 0

        pid_byte, pid_errors = self._decode_byte(data, pid_start_idx, bit_period, half_bit)

        frame_id = pid_byte & 0x3F
        parity = (pid_byte >> 6) & 0x03
        expected_parity = self._compute_parity(frame_id)

        if parity != expected_parity:
            pid_errors.append(f"Parity error: {parity} (expected {expected_parity})")

        return pid_byte, pid_errors, frame_id

    def _decode_data_fields(
        self,
        data: NDArray[np.bool_],
        sync_start_idx: int,
        bit_period: float,
        half_bit: float,
        data_length: int,
    ) -> tuple[list[int], list[str]]:
        """Decode all data bytes."""
        data_bytes = []
        data_errors = []
        data_start_idx = int(sync_start_idx + 20 * bit_period)

        for i in range(data_length):
            byte_start_idx = int(data_start_idx + i * 10 * bit_period)
            if byte_start_idx >= len(data):
                break

            byte_val, byte_errors = self._decode_byte(data, byte_start_idx, bit_period, half_bit)
            data_bytes.append(byte_val)
            data_errors.extend(byte_errors)

        return data_bytes, data_errors

    def _decode_checksum_field(
        self,
        data: NDArray[np.bool_],
        sync_start_idx: int,
        bit_period: float,
        half_bit: float,
        data_length: int,
        frame_id: int,
        data_bytes: list[int],
    ) -> tuple[int, list[str]]:
        """Decode and validate checksum field."""
        data_start_idx = int(sync_start_idx + 20 * bit_period)
        checksum_start_idx = int(data_start_idx + data_length * 10 * bit_period)

        if checksum_start_idx >= len(data):
            return 0, ["Missing checksum"]

        checksum_byte, checksum_errors = self._decode_byte(
            data, checksum_start_idx, bit_period, half_bit
        )

        expected_checksum = self._compute_checksum(frame_id, data_bytes)
        if checksum_byte != expected_checksum:
            checksum_errors.append(
                f"Checksum error: 0x{checksum_byte:02X} (expected 0x{expected_checksum:02X})"
            )

        return checksum_byte, checksum_errors

    def _create_lin_packet(
        self,
        break_start: int,
        sync_start_idx: int,
        bit_period: float,
        data_length: int,
        sample_rate: float,
        sync_errors: list[str],
        pid_errors: list[str],
        data_errors: list[str],
        checksum_errors: list[str],
        frame_num: int,
        frame_id: int,
        pid_byte: int,
        checksum_byte: int,
        data_bytes: list[int],
    ) -> ProtocolPacket:
        """Create LIN protocol packet from decoded fields."""
        data_start_idx = int(sync_start_idx + 20 * bit_period)
        checksum_start_idx = int(data_start_idx + data_length * 10 * bit_period)

        start_time = break_start / sample_rate
        end_time = (checksum_start_idx + 10 * bit_period) / sample_rate

        errors = sync_errors + pid_errors + data_errors + checksum_errors

        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.PACKETS,
            f"ID: 0x{frame_id:02X}",
            data=bytes(data_bytes),
        )

        annotations = {
            "frame_num": frame_num,
            "frame_id": frame_id,
            "pid": pid_byte,
            "data_length": data_length,
            "checksum": checksum_byte,
            "version": self._version.value,
        }

        return ProtocolPacket(
            timestamp=start_time,
            protocol="lin",
            data=bytes(data_bytes),
            annotations=annotations,
            errors=errors,
        )

    def _find_break_field(
        self,
        data: NDArray[np.bool_],
        start_idx: int,
        bit_period: float,
    ) -> int | None:
        """Find LIN break field (dominant for >= 13 bits).

        Args:
            data: Digital data array.
            start_idx: Index to start searching.
            bit_period: Bit period in samples.

        Returns:
            Index of break field start, or None if not found.
        """
        # Use a slightly smaller threshold to account for rounding
        # LIN spec requires >= 13 bit times, use 12.5 to be tolerant
        min_break_samples = int(12.5 * bit_period)

        idx = start_idx
        while idx < len(data) - min_break_samples:
            # Look for recessive-to-dominant transition
            if idx > 0 and data[idx - 1] and not data[idx]:
                # Check if dominant for at least 12.5 bit periods
                dominant_length = 0
                check_idx = idx
                while check_idx < len(data) and not data[check_idx]:
                    dominant_length += 1
                    check_idx += 1

                if dominant_length >= min_break_samples:
                    return idx

            idx += 1

        return None

    def _decode_byte(
        self,
        data: NDArray[np.bool_],
        start_idx: int,
        bit_period: float,
        half_bit: float,
    ) -> tuple[int, list[str]]:
        """Decode UART-style byte (1 start, 8 data, 1 stop).

        Args:
            data: Digital data array.
            start_idx: Start index (at start bit).
            bit_period: Bit period in samples.
            half_bit: Half bit period in samples.

        Returns:
            (byte_value, errors) tuple.
        """
        errors = []

        # Sample at center of each bit
        sample_points = []
        for bit_num in range(10):  # Start + 8 data + stop
            sample_idx = int(start_idx + half_bit + bit_num * bit_period)
            if sample_idx < len(data):
                sample_points.append(sample_idx)

        if len(sample_points) < 10:
            return 0, ["Incomplete byte"]

        # Verify start bit (should be 0)
        if data[sample_points[0]]:
            errors.append("Invalid start bit")

        # Extract data bits (LSB first)
        byte_val = 0
        for i in range(8):
            bit = 1 if data[sample_points[1 + i]] else 0
            byte_val |= bit << i

        # Verify stop bit (should be 1)
        if not data[sample_points[9]]:
            errors.append("Framing error")

        return byte_val, errors

    def _compute_parity(self, frame_id: int) -> int:
        """Compute LIN 2.x protected identifier parity.

        Args:
            frame_id: 6-bit frame identifier.

        Returns:
            2-bit parity value.
        """
        # Extract ID bits
        id0 = (frame_id >> 0) & 1
        id1 = (frame_id >> 1) & 1
        id2 = (frame_id >> 2) & 1
        id3 = (frame_id >> 3) & 1
        id4 = (frame_id >> 4) & 1
        id5 = (frame_id >> 5) & 1

        # P0 = ID0 ^ ID1 ^ ID2 ^ ID4
        p0 = id0 ^ id1 ^ id2 ^ id4

        # P1 = !(ID1 ^ ID3 ^ ID4 ^ ID5)
        p1 = (id1 ^ id3 ^ id4 ^ id5) ^ 1

        return (p1 << 1) | p0

    def _get_data_length(self, frame_id: int) -> int:
        """Get data length for frame ID.

        Args:
            frame_id: Frame identifier.

        Returns:
            Data length in bytes (0-8).
        """
        # Standard frame IDs have predefined lengths
        # For simplicity, assume 8 bytes (can be configured per application)
        return 8

    def _compute_checksum(self, frame_id: int, data_bytes: list[int]) -> int:
        """Compute LIN checksum.

        Args:
            frame_id: Frame identifier.
            data_bytes: Data bytes.

        Returns:
            Checksum byte.
        """
        if self._version == LINVersion.LIN_1X:
            # Classic checksum: sum of data bytes
            checksum = sum(data_bytes)
        else:
            # Enhanced checksum: sum of PID + data bytes
            pid = frame_id | (self._compute_parity(frame_id) << 6)
            checksum = pid + sum(data_bytes)

        # Handle carries
        while checksum > 0xFF:
            checksum = (checksum & 0xFF) + (checksum >> 8)

        # Invert
        return (~checksum) & 0xFF


def decode_lin(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    baudrate: int = 19200,
    version: Literal["1.x", "2.x"] = "2.x",
) -> list[ProtocolPacket]:
    """Convenience function to decode LIN frames.

    Args:
        data: LIN bus signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        baudrate: Baud rate (9600, 19200, 20000).
        version: LIN version ("1.x" or "2.x").

    Returns:
        List of decoded LIN frames.

    Example:
        >>> packets = decode_lin(signal, sample_rate=1e6, baudrate=19200)
        >>> for pkt in packets:
        ...     print(f"ID: 0x{pkt.annotations['frame_id']:02X}")
    """
    decoder = LINDecoder(baudrate=baudrate, version=version)
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        from oscura.core.types import TraceMetadata

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = DigitalTrace(data=data, metadata=metadata)
        return list(decoder.decode(trace))


__all__ = ["LINDecoder", "LINVersion", "decode_lin"]
