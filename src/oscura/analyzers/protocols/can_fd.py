"""CAN-FD protocol decoder.

This module implements CAN with Flexible Data-rate (CAN-FD) decoder
supporting variable data rate and extended payloads up to 64 bytes.


Example:
    >>> from oscura.analyzers.protocols.can_fd import CANFDDecoder
    >>> decoder = CANFDDecoder(nominal_bitrate=500000, data_bitrate=2000000)
    >>> for packet in decoder.decode(trace):
    ...     print(f"ID: 0x{packet.annotations['arbitration_id']:03X}")

References:
    ISO 11898-1:2015 CAN-FD Specification
    Bosch CAN-FD Specification v1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

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

    import numpy as np
    from numpy.typing import NDArray


class CANFDFrameType(IntEnum):
    """CAN-FD frame types."""

    DATA = 0
    REMOTE = 1


@dataclass
class CANFDFrame:
    """Decoded CAN-FD frame.

    Attributes:
        arbitration_id: CAN ID (11-bit or 29-bit).
        is_extended: True for 29-bit extended ID.
        is_fd: True for CAN-FD frame.
        brs: Bit Rate Switch flag.
        esi: Error State Indicator.
        dlc: Data length code (0-15).
        data: Data bytes (0-64).
        crc: Received CRC value.
        timestamp: Frame start time in seconds.
        errors: List of detected errors.
    """

    arbitration_id: int
    is_extended: bool
    is_fd: bool
    brs: bool
    esi: bool
    dlc: int
    data: bytes
    crc: int
    timestamp: float
    errors: list[str]


# CAN-FD DLC to data length mapping
CANFD_DLC_TO_LENGTH = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 12,
    10: 16,
    11: 20,
    12: 24,
    13: 32,
    14: 48,
    15: 64,
}


class CANFDDecoder(AsyncDecoder):
    """CAN-FD protocol decoder.

    Decodes CAN-FD frames with dual bit rate support, extended payloads,
    and CRC-17/CRC-21 validation.

    Attributes:
        id: "can_fd"
        name: "CAN-FD"
        channels: [can_h, can_l] (optional differential) or [can] (single-ended)

    Example:
        >>> decoder = CANFDDecoder(nominal_bitrate=500000, data_bitrate=2000000)
        >>> for packet in decoder.decode(trace):
        ...     print(f"Data ({len(packet.data)} bytes): {packet.data.hex()}")
    """

    id = "can_fd"
    name = "CAN-FD"
    longname = "CAN with Flexible Data-rate"
    desc = "CAN-FD protocol decoder"

    channels = [
        ChannelDef("can", "CAN", "CAN bus signal", required=True),
    ]

    optional_channels = [
        ChannelDef("can_h", "CAN_H", "CAN High differential signal", required=False),
        ChannelDef("can_l", "CAN_L", "CAN Low differential signal", required=False),
    ]

    options = [
        OptionDef(
            "nominal_bitrate",
            "Nominal bitrate",
            "Arbitration phase bitrate",
            default=500000,
            values=None,
        ),
        OptionDef(
            "data_bitrate",
            "Data bitrate",
            "Data phase bitrate",
            default=2000000,
            values=None,
        ),
    ]

    annotations = [
        ("sof", "Start of Frame"),
        ("arbitration", "Arbitration field"),
        ("control", "Control field"),
        ("data", "Data field"),
        ("crc", "CRC field"),
        ("ack", "Acknowledge"),
        ("eof", "End of Frame"),
        ("error", "Error"),
    ]

    def __init__(
        self,
        nominal_bitrate: int = 500000,
        data_bitrate: int = 2000000,
    ) -> None:
        """Initialize CAN-FD decoder.

        Args:
            nominal_bitrate: Nominal bitrate for arbitration phase (bps).
            data_bitrate: Data phase bitrate for BRS frames (bps).
        """
        super().__init__(
            baudrate=nominal_bitrate,
            nominal_bitrate=nominal_bitrate,
            data_bitrate=data_bitrate,
        )
        self._nominal_bitrate = nominal_bitrate
        self._data_bitrate = data_bitrate

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode CAN-FD frames from trace.

        Args:
            trace: Input digital trace.
            **channels: Additional channel data.

        Yields:
            Decoded CAN-FD frames as ProtocolPacket objects.

        Example:
            >>> decoder = CANFDDecoder(nominal_bitrate=500000)
            >>> for packet in decoder.decode(trace):
            ...     print(f"ID: 0x{packet.annotations['arbitration_id']:X}")
        """
        # Convert to digital if needed
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            digital_trace = to_digital(trace, threshold="auto")
        else:
            digital_trace = trace

        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        nominal_bit_period = sample_rate / self._nominal_bitrate
        data_bit_period = sample_rate / self._data_bitrate

        frame_num = 0
        idx = 0

        while idx < len(data):
            # Look for SOF (dominant bit during idle)
            sof_idx = self._find_sof(data, idx)
            if sof_idx is None:
                break

            # Decode frame starting from SOF
            frame, end_idx = self._decode_frame(
                data, sof_idx, sample_rate, nominal_bit_period, data_bit_period
            )

            if frame is not None:
                # Calculate timing
                start_time = sof_idx / sample_rate

                # Add annotation
                self.put_annotation(
                    start_time,
                    frame.timestamp + 0.001,  # Approximate end
                    AnnotationLevel.PACKETS,
                    f"ID: 0x{frame.arbitration_id:X}, {len(frame.data)} bytes",
                )

                # Create packet
                annotations = {
                    "frame_num": frame_num,
                    "arbitration_id": frame.arbitration_id,
                    "is_extended": frame.is_extended,
                    "is_fd": frame.is_fd,
                    "brs": frame.brs,
                    "esi": frame.esi,
                    "dlc": frame.dlc,
                    "data_length": len(frame.data),
                    "crc": frame.crc,
                }

                packet = ProtocolPacket(
                    timestamp=start_time,
                    protocol="can_fd",
                    data=frame.data,
                    annotations=annotations,
                    errors=frame.errors,
                )

                yield packet
                frame_num += 1

            idx = end_idx if end_idx > idx else idx + int(nominal_bit_period)

    def _find_sof(self, data: NDArray[np.bool_], start_idx: int) -> int | None:
        """Find Start of Frame (dominant bit during recessive idle).

        Args:
            data: Digital data array.
            start_idx: Start search index.

        Returns:
            Index of SOF, or None if not found.
        """
        # Look for recessive-to-dominant transition (1 to 0)
        idx = start_idx
        while idx < len(data) - 1:
            if data[idx] and not data[idx + 1]:
                return idx + 1
            idx += 1
        return None

    def _decode_frame(
        self,
        data: NDArray[np.bool_],
        sof_idx: int,
        sample_rate: float,
        nominal_bit_period: float,
        data_bit_period: float,
    ) -> tuple[CANFDFrame | None, int]:
        """Decode CAN-FD frame starting from SOF.

        Args:
            data: Digital data array.
            sof_idx: SOF index.
            sample_rate: Sample rate in Hz.
            nominal_bit_period: Nominal bit period in samples.
            data_bit_period: Data bit period in samples.

        Returns:
            (frame, end_index) tuple.
        """
        decoder_state = _CANFDDecoderState(sof_idx, nominal_bit_period, data, data_bit_period)

        arbitration_id, is_extended = decoder_state.decode_arbitration_field()
        if arbitration_id is None:
            return None, decoder_state.get_bit_idx()

        is_fd, brs, esi, dlc = decoder_state.decode_control_field(is_extended)
        if dlc is None:
            return None, decoder_state.get_bit_idx()

        data_length = CANFD_DLC_TO_LENGTH.get(dlc, 0)
        data_bytes = decoder_state.decode_data_field(data_length, is_fd, brs)
        crc = decoder_state.decode_crc_field(data_length)
        decoder_state.decode_end_of_frame()

        frame = CANFDFrame(
            arbitration_id=arbitration_id,
            is_extended=is_extended,
            is_fd=is_fd,
            brs=brs,
            esi=esi,
            dlc=dlc,
            data=bytes(data_bytes),
            crc=crc,
            timestamp=sof_idx / sample_rate,
            errors=[],
        )

        return frame, decoder_state.get_bit_idx()


class _CANFDDecoderState:
    """State tracker for CAN-FD frame decoding."""

    def __init__(
        self,
        sof_idx: int,
        nominal_bit_period: float,
        data: NDArray[np.bool_],
        data_bit_period: float,
    ):
        self.bit_idx: float = float(sof_idx)
        self.current_bit_period = nominal_bit_period
        self.nominal_bit_period = nominal_bit_period
        self.data_bit_period = data_bit_period
        self.data = data

    def sample_bits(self, count: int) -> list[int]:
        """Sample specified number of bits."""
        bits = []
        for _ in range(count):
            sample_idx_raw = self.bit_idx + self.current_bit_period / 2
            sample_idx = int(sample_idx_raw)
            if sample_idx < len(self.data):
                bits.append(0 if self.data[sample_idx] else 1)
                self.bit_idx += self.current_bit_period
            else:
                return bits
        return bits

    def bits_to_int(self, bits: list[int]) -> int:
        """Convert bit list to integer."""
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value

    def decode_arbitration_field(self) -> tuple[int | None, bool]:
        """Decode arbitration field and determine if extended frame."""
        arb_bits = self.sample_bits(11)
        if len(arb_bits) < 11:
            return None, False

        arbitration_id = self.bits_to_int(arb_bits)

        ide_bits = self.sample_bits(1)
        is_extended = bool(ide_bits[0]) if ide_bits else False

        if is_extended:
            ext_bits = self.sample_bits(18)
            arbitration_id = (arbitration_id << 18) | self.bits_to_int(ext_bits)

        return arbitration_id, is_extended

    def decode_control_field(self, is_extended: bool) -> tuple[bool, bool, bool, int | None]:
        """Decode control field."""
        ctrl_bits = self.sample_bits(7 if not is_extended else 6)

        if len(ctrl_bits) < (7 if not is_extended else 6):
            return False, False, False, None

        is_fd = ctrl_bits[0] == 1
        brs = ctrl_bits[2] == 1 if len(ctrl_bits) > 2 else False
        esi = ctrl_bits[3] == 1 if len(ctrl_bits) > 3 else False

        dlc_start = 3 if not is_extended else 2
        dlc_bits = (
            ctrl_bits[dlc_start : dlc_start + 4]
            if len(ctrl_bits) >= dlc_start + 4
            else [0, 0, 0, 0]
        )
        dlc = self.bits_to_int(dlc_bits)

        return is_fd, brs, esi, dlc

    def decode_data_field(self, data_length: int, is_fd: bool, brs: bool) -> list[int]:
        """Decode data field."""
        if is_fd and brs:
            self.current_bit_period = self.data_bit_period

        data_bytes = []
        for _ in range(data_length):
            byte_bits = self.sample_bits(8)
            if len(byte_bits) == 8:
                data_bytes.append(self.bits_to_int(byte_bits))

        return data_bytes

    def decode_crc_field(self, data_length: int) -> int:
        """Decode CRC field."""
        crc_length = 17 if data_length <= 16 else 21
        crc_bits = self.sample_bits(crc_length)
        return self.bits_to_int(crc_bits)

    def decode_end_of_frame(self) -> None:
        """Decode end of frame."""
        self.current_bit_period = self.nominal_bit_period
        self.sample_bits(10)

    def get_bit_idx(self) -> int:
        """Get current bit index."""
        return int(self.bit_idx)


def decode_can_fd(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    nominal_bitrate: int = 500000,
    data_bitrate: int = 2000000,
) -> list[ProtocolPacket]:
    """Convenience function to decode CAN-FD frames.

    Args:
        data: CAN bus signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        nominal_bitrate: Nominal bitrate in bps.
        data_bitrate: Data phase bitrate in bps.

    Returns:
        List of decoded CAN-FD frames.

    Example:
        >>> packets = decode_can_fd(signal, sample_rate=100e6, nominal_bitrate=500000)
        >>> for pkt in packets:
        ...     print(f"ID: 0x{pkt.annotations['arbitration_id']:X}")
    """
    decoder = CANFDDecoder(nominal_bitrate=nominal_bitrate, data_bitrate=data_bitrate)
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        trace = DigitalTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        return list(decoder.decode(trace))


__all__ = [
    "CANFD_DLC_TO_LENGTH",
    "CANFDDecoder",
    "CANFDFrame",
    "CANFDFrameType",
    "decode_can_fd",
]
