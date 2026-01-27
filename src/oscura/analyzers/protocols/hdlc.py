"""HDLC protocol decoder.

This module provides High-Level Data Link Control (HDLC) telecom protocol
decoding with bit stuffing, FCS validation, and field extraction.


Example:
    >>> from oscura.analyzers.protocols.hdlc import HDLCDecoder
    >>> decoder = HDLCDecoder()
    >>> for packet in decoder.decode(trace):
    ...     print(f"Address: 0x{packet.annotations['address']:02X}")

References:
    ISO/IEC 13239:2002 - HDLC Frame Structure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
from oscura.utils.bitwise import bits_to_byte

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray


class HDLCDecoder(AsyncDecoder):
    """HDLC protocol decoder.

    Decodes HDLC frames with flag detection, bit unstuffing,
    and FCS (Frame Check Sequence) validation using CRC-16 or CRC-32.

    Attributes:
        id: "hdlc"
        name: "HDLC"
        channels: [data] (required)

    Example:
        >>> decoder = HDLCDecoder(baudrate=1000000, fcs="crc16")
        >>> for packet in decoder.decode(trace):
        ...     print(f"Info: {packet.data.hex()}")
    """

    id = "hdlc"
    name = "HDLC"
    longname = "High-Level Data Link Control"
    desc = "HDLC telecom protocol decoder"

    channels = [
        ChannelDef("data", "DATA", "HDLC data line", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef("baudrate", "Baud rate", "Bits per second", default=1000000, values=None),
        OptionDef(
            "fcs",
            "FCS type",
            "Frame check sequence",
            default="crc16",
            values=["crc16", "crc32"],
        ),
    ]

    annotations = [
        ("flag", "Flag sequence"),
        ("address", "Address field"),
        ("control", "Control field"),
        ("info", "Information field"),
        ("fcs", "Frame check sequence"),
        ("error", "Error"),
    ]

    # HDLC flag pattern
    FLAG_PATTERN = 0b01111110  # 0x7E

    def __init__(
        self,
        baudrate: int = 1000000,
        fcs: Literal["crc16", "crc32"] = "crc16",
    ) -> None:
        """Initialize HDLC decoder.

        Args:
            baudrate: Baud rate in bps.
            fcs: FCS type ("crc16" or "crc32").
        """
        super().__init__(baudrate=baudrate, fcs=fcs)
        self._fcs = fcs
        self._fcs_bytes = 2 if fcs == "crc16" else 4

    def _process_frame_bytes(
        self,
        unstuffed_bits: list[int],
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """Convert unstuffed bits to bytes and extract frame fields."""
        field_bytes = []
        for i in range(0, len(unstuffed_bits), 8):
            if i + 8 <= len(unstuffed_bits):
                byte_val = bits_to_byte(unstuffed_bits[i : i + 8], lsb_first=True)
                field_bytes.append(byte_val)

        address = field_bytes[0]
        control = field_bytes[1]
        info_bytes = field_bytes[2 : -self._fcs_bytes]
        fcs_bytes = field_bytes[-self._fcs_bytes :]

        return field_bytes, address, control, info_bytes, fcs_bytes

    def _validate_fcs(
        self,
        frame_data: list[int],
        fcs_bytes: list[int],
    ) -> tuple[bool, list[str]]:
        """Validate Frame Check Sequence and return errors if any."""
        errors = []

        if self._fcs == "crc16":
            computed_fcs = self._crc16_ccitt(bytes(frame_data))
            received_fcs = (fcs_bytes[1] << 8) | fcs_bytes[0]
        else:
            computed_fcs = self._crc32(bytes(frame_data))
            received_fcs = (
                (fcs_bytes[3] << 24) | (fcs_bytes[2] << 16) | (fcs_bytes[1] << 8) | fcs_bytes[0]
            )

        if computed_fcs != received_fcs:
            errors.append("FCS mismatch")

        return computed_fcs == received_fcs, errors

    def _create_hdlc_packet(
        self,
        frame_num: int,
        address: int,
        control: int,
        info_bytes: list[int],
        errors: list[str],
        start_time: float,
    ) -> ProtocolPacket:
        """Create HDLC protocol packet from decoded frame."""
        annotations = {
            "frame_num": frame_num,
            "address": address,
            "control": control,
            "info_length": len(info_bytes),
            "fcs_type": self._fcs,
        }

        return ProtocolPacket(
            timestamp=start_time,
            protocol="hdlc",
            data=bytes(info_bytes),
            annotations=annotations,
            errors=errors,
        )

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode HDLC frames from trace.

        Args:
            trace: Input digital trace.
            **channels: Additional channel data.

        Yields:
            Decoded HDLC frames as ProtocolPacket objects.

        Example:
            >>> decoder = HDLCDecoder(baudrate=1000000)
            >>> for packet in decoder.decode(trace):
            ...     print(f"Address: 0x{packet.annotations['address']:02X}")
        """
        # Convert to digital if needed
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            digital_trace = to_digital(trace, threshold="auto")
        else:
            digital_trace = trace

        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        bit_period = sample_rate / self._baudrate

        # Extract bit stream
        bits = self._sample_bits(data, bit_period)

        # Find frames (between flag sequences)
        frame_num = 0
        idx = 0

        while idx < len(bits):
            # Look for opening flag
            flag_idx = self._find_flag(bits, idx)
            if flag_idx is None:
                break

            # Look for closing flag
            next_flag_idx = self._find_flag(bits, flag_idx + 8)
            if next_flag_idx is None:
                break

            # Extract frame bits (between flags, excluding flags)
            frame_bits = bits[flag_idx + 8 : next_flag_idx]

            if len(frame_bits) < 16:  # Minimum: address(8) + control(8)
                idx = next_flag_idx + 8
                continue

            # Bit unstuffing (remove 0 after five consecutive 1s)
            unstuffed_bits, stuff_errors = self._unstuff_bits(frame_bits)

            if len(unstuffed_bits) < 16 + self._fcs_bytes * 8:
                idx = next_flag_idx + 8
                continue

            # Extract and process frame fields
            field_bytes, address, control, info_bytes, fcs_bytes = self._process_frame_bytes(
                unstuffed_bits
            )

            if len(field_bytes) < 2 + self._fcs_bytes:
                idx = next_flag_idx + 8
                continue

            # Validate FCS
            errors = list(stuff_errors)
            frame_data = field_bytes[: -self._fcs_bytes]
            _, fcs_errors = self._validate_fcs(frame_data, fcs_bytes)
            errors.extend(fcs_errors)

            # Calculate timing
            start_time = (flag_idx * bit_period) / sample_rate
            end_time = ((next_flag_idx + 8) * bit_period) / sample_rate

            # Add annotation
            self.put_annotation(
                start_time,
                end_time,
                AnnotationLevel.PACKETS,
                f"Addr: 0x{address:02X}, Ctrl: 0x{control:02X}",
            )

            # Create and yield packet
            packet = self._create_hdlc_packet(
                frame_num, address, control, info_bytes, errors, start_time
            )
            yield packet

            frame_num += 1
            idx = next_flag_idx + 8

    def _sample_bits(
        self,
        data: NDArray[np.bool_],
        bit_period: float,
    ) -> list[int]:
        """Sample data at bit centers to extract bit stream.

        Args:
            data: Digital data array.
            bit_period: Bit period in samples.

        Returns:
            List of bit values.
        """
        bits = []
        idx = 0
        while idx < len(data):
            sample_idx = int(idx + bit_period / 2)
            if sample_idx < len(data):
                bits.append(1 if data[sample_idx] else 0)
            idx += bit_period  # type: ignore[assignment]
        return bits

    def _find_flag(self, bits: list[int], start_idx: int) -> int | None:
        """Find HDLC flag pattern (01111110).

        Args:
            bits: Bit stream.
            start_idx: Start search index.

        Returns:
            Index of flag start, or None if not found.
        """
        for i in range(start_idx, len(bits) - 7):
            if (
                bits[i] == 0
                and bits[i + 1] == 1
                and bits[i + 2] == 1
                and bits[i + 3] == 1
                and bits[i + 4] == 1
                and bits[i + 5] == 1
                and bits[i + 6] == 1
                and bits[i + 7] == 0
            ):
                return i
        return None

    def _unstuff_bits(self, bits: list[int]) -> tuple[list[int], list[str]]:
        """Remove bit stuffing (0 after five consecutive 1s).

        Args:
            bits: Stuffed bit stream.

        Returns:
            (unstuffed_bits, errors) tuple.
        """
        unstuffed = []
        errors = []  # type: ignore[var-annotated]
        ones_count = 0

        i = 0
        while i < len(bits):
            bit = bits[i]

            if bit == 1:
                ones_count += 1
                unstuffed.append(bit)
            elif ones_count == 5:
                # This is a stuff bit, skip it
                ones_count = 0
            else:
                unstuffed.append(bit)
                ones_count = 0

            i += 1

        return unstuffed, errors

    def _crc16_ccitt(self, data: bytes) -> int:
        """Compute CRC-16-CCITT.

        Args:
            data: Input data bytes.

        Returns:
            16-bit CRC.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = (crc << 1 ^ 4129) & 65535 if crc & 32768 else crc << 1 & 65535
        return crc ^ 0xFFFF

    def _crc32(self, data: bytes) -> int:
        """Compute CRC-32.

        Args:
            data: Input data bytes.

        Returns:
            32-bit CRC.
        """
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc >>= 1
        return crc ^ 0xFFFFFFFF

    def _bits_to_byte(self, bits: list[int]) -> int:
        """Convert bit list to byte value (LSB first).

        Wrapper for bitwise.bits_to_byte utility.

        Args:
            bits: List of bits (up to 8).

        Returns:
            Byte value (0-255).
        """
        return bits_to_byte(bits)


def decode_hdlc(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    baudrate: int = 1000000,
    fcs: Literal["crc16", "crc32"] = "crc16",
) -> list[ProtocolPacket]:
    """Convenience function to decode HDLC frames.

    Args:
        data: HDLC data signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        baudrate: Baud rate in bps.
        fcs: FCS type ("crc16" or "crc32").

    Returns:
        List of decoded HDLC frames.

    Example:
        >>> packets = decode_hdlc(signal, sample_rate=10e6, baudrate=1000000)
        >>> for pkt in packets:
        ...     print(f"Address: 0x{pkt.annotations['address']:02X}")
    """
    decoder = HDLCDecoder(baudrate=baudrate, fcs=fcs)
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        trace = DigitalTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        return list(decoder.decode(trace))


__all__ = ["HDLCDecoder", "decode_hdlc"]
