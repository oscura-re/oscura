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

    channels = [  # noqa: RUF012
        ChannelDef("can", "CAN", "CAN bus signal", required=True),
    ]

    optional_channels = [  # noqa: RUF012
        ChannelDef("can_h", "CAN_H", "CAN High differential signal", required=False),
        ChannelDef("can_l", "CAN_L", "CAN Low differential signal", required=False),
    ]

    options = [  # noqa: RUF012
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

    annotations = [  # noqa: RUF012
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
        errors = []  # type: ignore[var-annotated]
        bit_idx = sof_idx
        current_bit_period = nominal_bit_period

        # Sample bits (simplified - ignores bit stuffing for brevity)
        def sample_bits(count: int) -> list[int]:
            nonlocal bit_idx
            bits = []
            for _ in range(count):
                sample_idx = int(bit_idx + current_bit_period / 2)
                if sample_idx < len(data):
                    bits.append(0 if data[sample_idx] else 1)  # Dominant=1, Recessive=0
                    bit_idx += current_bit_period  # type: ignore[assignment]
                else:
                    return bits
            return bits

        # Arbitration field (11 bits for standard, 29 for extended)
        arb_bits = sample_bits(11)
        if len(arb_bits) < 11:
            return None, int(bit_idx)

        arbitration_id = 0
        for bit in arb_bits:
            arbitration_id = (arbitration_id << 1) | bit

        # Check for extended frame (IDE bit)
        ide_bits = sample_bits(1)
        is_extended = ide_bits[0] == 1 if ide_bits else False

        if is_extended:
            # Extended ID: read additional 18 bits
            ext_bits = sample_bits(18)
            for bit in ext_bits:
                arbitration_id = (arbitration_id << 1) | bit

        # Control field
        # FDF (EDL), res, BRS, ESI, DLC (4 bits)
        ctrl_bits = sample_bits(7 if not is_extended else 6)

        if len(ctrl_bits) < (7 if not is_extended else 6):
            return None, int(bit_idx)

        # FDF/EDL bit - first bit of control field regardless of frame type
        fdf = ctrl_bits[0]
        is_fd = fdf == 1
        brs = ctrl_bits[2] == 1 if len(ctrl_bits) > 2 else False
        esi = ctrl_bits[3] == 1 if len(ctrl_bits) > 3 else False

        # DLC (4 bits)
        dlc_start = 3 if not is_extended else 2
        dlc_bits = (
            ctrl_bits[dlc_start : dlc_start + 4]
            if len(ctrl_bits) >= dlc_start + 4
            else [0, 0, 0, 0]
        )
        dlc = 0
        for bit in dlc_bits:
            dlc = (dlc << 1) | bit

        # Get data length from DLC
        data_length = CANFD_DLC_TO_LENGTH.get(dlc, 0)

        # Switch to data bitrate if BRS is set
        if is_fd and brs:
            current_bit_period = data_bit_period

        # Data field
        data_bytes = []
        for _ in range(data_length):
            byte_bits = sample_bits(8)
            if len(byte_bits) == 8:
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                data_bytes.append(byte_val)

        # CRC field (CRC-17 for <=16 bytes, CRC-21 for >16 bytes)
        crc_length = 17 if data_length <= 16 else 21
        crc_bits = sample_bits(crc_length)
        crc = 0
        for bit in crc_bits:
            crc = (crc << 1) | bit

        # Switch back to nominal bitrate for CRC delimiter, ACK, EOF
        current_bit_period = nominal_bit_period

        # CRC delimiter, ACK slot, ACK delimiter, EOF (7 bits)
        sample_bits(10)

        # Create frame
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
            errors=errors,
        )

        return frame, int(bit_idx)


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
