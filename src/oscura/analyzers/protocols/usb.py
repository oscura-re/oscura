"""USB protocol decoder.

This module provides USB Low Speed (1.5 Mbps) and Full Speed (12 Mbps)
protocol decoding with NRZI encoding, bit stuffing, and CRC validation.


Example:
    >>> from oscura.analyzers.protocols.usb import USBDecoder
    >>> decoder = USBDecoder(speed="full")
    >>> for packet in decoder.decode(dp=dp, dm=dm):
    ...     print(f"PID: {packet.annotations['pid_name']}")

References:
    USB 2.0 Specification (usb.org)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    ChannelDef,
    OptionDef,
    SyncDecoder,
)
from oscura.core.types import DigitalTrace, ProtocolPacket
from oscura.utils.bitwise import bits_to_byte

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray


class USBSpeed(Enum):
    """USB speed modes."""

    LOW_SPEED = 1_500_000  # 1.5 Mbps
    FULL_SPEED = 12_000_000  # 12 Mbps


class USBPID(Enum):
    """USB Packet Identifiers."""

    # Token PIDs
    OUT = 0b0001
    IN = 0b1001
    SOF = 0b0101
    SETUP = 0b1101
    # Data PIDs
    DATA0 = 0b0011
    DATA1 = 0b1011
    DATA2 = 0b0111
    MDATA = 0b1111
    # Handshake PIDs
    ACK = 0b0010
    NAK = 0b1010
    STALL = 0b1110
    NYET = 0b0110
    # Special PIDs
    PRE = 0b1100
    ERR = 0b1100
    SPLIT = 0b1000
    PING = 0b0100


# PID names for display
PID_NAMES = {
    0b0001: "OUT",
    0b1001: "IN",
    0b0101: "SOF",
    0b1101: "SETUP",
    0b0011: "DATA0",
    0b1011: "DATA1",
    0b0111: "DATA2",
    0b1111: "MDATA",
    0b0010: "ACK",
    0b1010: "NAK",
    0b1110: "STALL",
    0b0110: "NYET",
    0b1100: "PRE/ERR",
    0b1000: "SPLIT",
    0b0100: "PING",
}


class USBDecoder(SyncDecoder):
    """USB protocol decoder.

    Decodes USB Low Speed and Full Speed transactions including
    NRZI decoding, bit unstuffing, and CRC validation.

    Attributes:
        id: "usb"
        name: "USB"
        channels: [dp, dm] (required)

    Example:
        >>> decoder = USBDecoder(speed="full")
        >>> for packet in decoder.decode(dp=dp, dm=dm, sample_rate=100e6):
        ...     print(f"PID: {packet.annotations['pid_name']}")
    """

    id = "usb"
    name = "USB"
    longname = "Universal Serial Bus"
    desc = "USB Low/Full Speed protocol decoder"

    channels = [
        ChannelDef("dp", "D+", "USB D+ signal", required=True),
        ChannelDef("dm", "D-", "USB D- signal", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef("speed", "Speed", "USB speed", default="full", values=["low", "full"]),
    ]

    annotations = [
        ("sync", "SYNC field"),
        ("pid", "Packet ID"),
        ("data", "Data payload"),
        ("crc", "CRC field"),
        ("eop", "End of Packet"),
        ("error", "Error"),
    ]

    def __init__(
        self,
        speed: Literal["low", "full"] = "full",
    ) -> None:
        """Initialize USB decoder.

        Args:
            speed: USB speed ("low" or "full").
        """
        super().__init__(speed=speed)
        self._speed = USBSpeed.LOW_SPEED if speed == "low" else USBSpeed.FULL_SPEED

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        dp: NDArray[np.bool_] | None = None,
        dm: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode USB packets.

        Args:
            trace: Optional primary trace.
            dp: D+ signal.
            dm: D- signal.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded USB packets as ProtocolPacket objects.

        Example:
            >>> decoder = USBDecoder(speed="full")
            >>> for pkt in decoder.decode(dp=dp, dm=dm, sample_rate=100e6):
            ...     print(f"Address: {pkt.annotations.get('address', 'N/A')}")
        """
        if dp is None or dm is None:
            return

        diff_signal, se0, bit_period = self._prepare_signals(dp, dm, sample_rate)

        trans_num = 0
        idx = 0

        while idx < len(diff_signal):
            nrzi_start = self._find_sync_pattern(diff_signal, idx, bit_period)
            if nrzi_start is None:
                break

            packet_bits, bit_errors = self._extract_packet_bits(
                diff_signal, nrzi_start, bit_period, se0
            )

            if len(packet_bits) < 16:  # Minimum: SYNC(8) + PID(8)
                idx = nrzi_start + int(bit_period)
                continue

            pid_value, pid_name, errors = self._parse_pid(packet_bits, bit_errors)

            payload_bits = packet_bits[16:]  # Skip SYNC(8) + PID(8)
            payload_bytes, annotations = self._parse_payload(
                pid_value, pid_name, payload_bits, trans_num, errors
            )

            packet = self._build_usb_packet(
                nrzi_start,
                packet_bits,
                bit_period,
                sample_rate,
                pid_name,
                payload_bytes,
                annotations,
                errors,
            )

            yield packet

            trans_num += 1
            idx = int(nrzi_start + len(packet_bits) * bit_period)

    def _prepare_signals(
        self, dp: NDArray[np.bool_], dm: NDArray[np.bool_], sample_rate: float
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float]:
        """Prepare differential and SE0 signals.

        Args:
            dp: D+ signal.
            dm: D- signal.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (differential_signal, se0_signal, bit_period).
        """
        n_samples = min(len(dp), len(dm))
        dp = dp[:n_samples]
        dm = dm[:n_samples]

        # Decode differential signal
        if self._speed == USBSpeed.LOW_SPEED:
            diff_signal = ~dp & dm  # Low speed: J = D-, K = D+
        else:
            diff_signal = dp & ~dm  # Full speed: J = D+, K = D-

        se0 = ~dp & ~dm  # SE0: both D+ and D- are 0
        bit_period = sample_rate / self._speed.value

        return diff_signal, se0, bit_period

    def _parse_pid(
        self, packet_bits: list[int], bit_errors: list[str]
    ) -> tuple[int, str, list[str]]:
        """Parse and validate PID field.

        Args:
            packet_bits: Full packet bits including SYNC.
            bit_errors: Bit-level errors from extraction.

        Returns:
            Tuple of (pid_value, pid_name, errors).
        """
        data_bits = packet_bits[8:]  # Skip SYNC
        errors = list(bit_errors)

        if len(data_bits) < 8:
            return 0, "INVALID", errors

        pid_byte = bits_to_byte(data_bits[:8])
        pid_value = pid_byte & 0x0F
        pid_check = (pid_byte >> 4) & 0x0F

        # Validate PID (upper 4 bits should be complement of lower 4 bits)
        if pid_value ^ pid_check != 0x0F:
            errors.append("PID check failed")

        pid_name = PID_NAMES.get(pid_value, f"UNKNOWN(0x{pid_value:X})")

        return pid_value, pid_name, errors

    def _parse_payload(
        self,
        pid_value: int,
        pid_name: str,
        payload_bits: list[int],
        trans_num: int,
        errors: list[str],
    ) -> tuple[list[int], dict[str, Any]]:
        """Parse payload based on PID type.

        Args:
            pid_value: PID value.
            pid_name: PID name string.
            payload_bits: Payload bits after PID.
            trans_num: Transaction number.
            errors: Error list to append to.

        Returns:
            Tuple of (payload_bytes, annotations).
        """
        annotations = {
            "transaction_num": trans_num,
            "pid_value": pid_value,
            "pid_name": pid_name,
        }
        payload_bytes = []

        # Token packets: OUT, IN, SETUP
        if pid_value in [0b0001, 0b1001, 0b1101]:
            self._parse_token_payload(payload_bits, annotations, errors)
        # SOF packet
        elif pid_value == 0b0101:
            self._parse_sof_payload(payload_bits, annotations)
        # Data packets: DATA0, DATA1, DATA2, MDATA
        elif pid_value in [0b0011, 0b1011, 0b0111, 0b1111]:
            payload_bytes = self._parse_data_payload(payload_bits, annotations)

        return payload_bytes, annotations

    def _parse_token_payload(
        self, payload_bits: list[int], annotations: dict[str, Any], errors: list[str]
    ) -> None:
        """Parse token packet payload (address + endpoint + CRC5).

        Args:
            payload_bits: Payload bits.
            annotations: Annotations dict to update.
            errors: Error list to append to.
        """
        if len(payload_bits) >= 16:  # 11-bit (addr+endp) + 5-bit CRC
            addr_endp = self._bits_to_value(payload_bits[:11])
            address = addr_endp & 0x7F
            endpoint = (addr_endp >> 7) & 0x0F
            crc5 = self._bits_to_value(payload_bits[11:16])

            # Validate CRC5
            expected_crc5 = self._crc5(addr_endp)
            if crc5 != expected_crc5:
                errors.append("CRC5 error")

            annotations["address"] = address
            annotations["endpoint"] = endpoint

    def _parse_sof_payload(self, payload_bits: list[int], annotations: dict[str, Any]) -> None:
        """Parse SOF packet payload (frame number + CRC5).

        Args:
            payload_bits: Payload bits.
            annotations: Annotations dict to update.
        """
        if len(payload_bits) >= 16:  # 11-bit frame number + 5-bit CRC
            frame_num = self._bits_to_value(payload_bits[:11])
            annotations["frame_number"] = frame_num

    def _parse_data_payload(
        self, payload_bits: list[int], annotations: dict[str, Any]
    ) -> list[int]:
        """Parse data packet payload (data + CRC16).

        Args:
            payload_bits: Payload bits.
            annotations: Annotations dict to update.

        Returns:
            List of data bytes.
        """
        payload_bytes = []

        if len(payload_bits) >= 16:  # At least CRC16
            data_bit_count = len(payload_bits) - 16
            if data_bit_count >= 0:
                for i in range(0, data_bit_count, 8):
                    if i + 8 <= data_bit_count:
                        byte_val = bits_to_byte(payload_bits[i : i + 8])
                        payload_bytes.append(byte_val)

                annotations["data_length"] = len(payload_bytes)

        return payload_bytes

    def _build_usb_packet(
        self,
        start_idx: int,
        packet_bits: list[int],
        bit_period: float,
        sample_rate: float,
        pid_name: str,
        payload_bytes: list[int],
        annotations: dict[str, Any],
        errors: list[str],
    ) -> ProtocolPacket:
        """Build USB protocol packet.

        Args:
            start_idx: Packet start index.
            packet_bits: All packet bits.
            bit_period: Bit period in samples.
            sample_rate: Sample rate in Hz.
            pid_name: PID name string.
            payload_bytes: Payload data bytes.
            annotations: Packet annotations.
            errors: Error list.

        Returns:
            Protocol packet.
        """
        start_time = start_idx / sample_rate
        end_time = (start_idx + len(packet_bits) * bit_period) / sample_rate

        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.PACKETS,
            f"{pid_name}",
        )

        return ProtocolPacket(
            timestamp=start_time,
            protocol="usb",
            data=bytes(payload_bytes),
            annotations=annotations,
            errors=errors,
        )

    def _find_sync_pattern(
        self,
        signal: NDArray[np.bool_],
        start_idx: int,
        bit_period: float,
    ) -> int | None:
        """Find USB SYNC pattern (KJKJKJKK in differential).

        Args:
            signal: Differential signal.
            start_idx: Start search index.
            bit_period: Bit period in samples.

        Returns:
            Index of bit center of first SYNC bit, or None if not found.
        """
        # SYNC pattern is 00000001 in data
        # In NRZI: 0 = transition, 1 = no transition
        # So SYNC has 6 transitions in first 7 bits, then 1 non-transition
        # Look for alternating pattern

        min_transitions = 5
        idx = start_idx
        int(bit_period)

        while idx < len(signal) - int(9 * bit_period):
            # Sample 8 consecutive bits at bit centers
            # Bit centers are at: idx, idx+period, idx+2*period, ...
            samples = []
            for i in range(8):
                sample_idx = int(idx + i * bit_period)
                if sample_idx >= len(signal):
                    break
                samples.append(signal[sample_idx])

            if len(samples) < 8:
                break

            # Count transitions
            trans_count = sum(1 for i in range(7) if samples[i] != samples[i + 1])

            # SYNC should have at least 5-6 transitions
            if trans_count >= min_transitions:
                return idx  # Return first bit center

            idx += int(bit_period / 4)  # Scan at quarter-bit resolution

        return None

    def _extract_packet_bits(
        self,
        signal: NDArray[np.bool_],
        start_idx: int,
        bit_period: float,
        se0: NDArray[np.bool_],
    ) -> tuple[list[int], list[str]]:
        """Extract and decode packet bits with NRZI and unstuffing.

        Args:
            signal: NRZI-encoded differential signal.
            start_idx: Packet start index (bit center of first bit).
            bit_period: Bit period in samples.
            se0: SE0 detection array.

        Returns:
            (bits, errors) tuple.
        """
        bits = []
        errors = []  # type: ignore[var-annotated]
        stuff_count = 0

        max_bits = 1024  # Prevent infinite loops

        # Track previous signal value for NRZI decoding
        prev_sample_idx = max(0, int(start_idx - bit_period))
        prev_val = signal[prev_sample_idx]

        for bit_num in range(max_bits):
            sample_idx = int(start_idx + bit_num * bit_period)
            if sample_idx >= len(signal):
                break

            # Check for EOP (SE0)
            if se0[sample_idx]:
                break

            curr_val = signal[sample_idx]

            # NRZI decode: no transition = 1, transition = 0
            if curr_val == prev_val:
                bit = 1
            else:
                bit = 0

            # Bit unstuffing: after six consecutive 1s, the next 0 is a stuff bit
            if bit == 1:
                stuff_count += 1
                bits.append(bit)
            elif stuff_count == 6:
                # This is a stuff bit (should be 0), skip it
                stuff_count = 0
                # Don't append stuff bit
            else:
                # Normal 0 bit
                stuff_count = 0
                bits.append(bit)

            # Update for next bit
            prev_val = curr_val

        return bits, errors

    def _bits_to_value(self, bits: list[int]) -> int:
        """Convert bits to integer (LSB first).

        Args:
            bits: List of bits.

        Returns:
            Integer value.
        """
        value = 0
        for i, bit in enumerate(bits):
            value |= bit << i
        return value

    def _bits_to_byte(self, bits: list[int]) -> int:
        """Convert 8 bits to byte value (LSB first).

        Alias for _bits_to_value for backward compatibility.

        Args:
            bits: List of 8 bits.

        Returns:
            Byte value (0-255).
        """
        return self._bits_to_value(bits[:8])

    def _crc5(self, data: int) -> int:
        """Compute USB CRC5.

        Args:
            data: 11-bit data value.

        Returns:
            5-bit CRC.
        """
        # CRC-5-USB polynomial: x^5 + x^2 + 1 (0x05)
        crc = 0x1F
        for i in range(11):
            bit = (data >> i) & 1
            if (crc & 1) ^ bit:
                crc = ((crc >> 1) ^ 0x14) & 0x1F
            else:
                crc >>= 1
        return crc ^ 0x1F


def decode_usb(
    dp: NDArray[np.bool_],
    dm: NDArray[np.bool_],
    sample_rate: float = 1.0,
    speed: Literal["low", "full"] = "full",
) -> list[ProtocolPacket]:
    """Convenience function to decode USB packets.

    Args:
        dp: D+ signal.
        dm: D- signal.
        sample_rate: Sample rate in Hz.
        speed: USB speed ("low" or "full").

    Returns:
        List of decoded USB packets.

    Example:
        >>> packets = decode_usb(dp, dm, sample_rate=100e6, speed="full")
        >>> for pkt in packets:
        ...     print(f"PID: {pkt.annotations['pid_name']}")
    """
    decoder = USBDecoder(speed=speed)
    return list(decoder.decode(dp=dp, dm=dm, sample_rate=sample_rate))


__all__ = ["PID_NAMES", "USBPID", "USBDecoder", "USBSpeed", "decode_usb"]
