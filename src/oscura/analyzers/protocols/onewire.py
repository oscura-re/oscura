"""Dallas/Maxim 1-Wire protocol decoder.

This module provides a 1-Wire protocol decoder for temperature sensors,
EEPROMs, and other 1-Wire devices with ROM command decoding.


Example:
    >>> from oscura.analyzers.protocols.onewire import OneWireDecoder
    >>> decoder = OneWireDecoder()
    >>> for packet in decoder.decode(trace):
    ...     print(f"ROM: {packet.annotations['rom_id']}")

References:
    Dallas/Maxim 1-Wire Protocol
    DS18B20 Datasheet
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

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


class OneWireMode(Enum):
    """1-Wire speed modes."""

    STANDARD = "standard"  # Standard speed (15.4 kbps)
    OVERDRIVE = "overdrive"  # Overdrive speed (~142 kbps)


class OneWireROMCommand(Enum):
    """1-Wire ROM commands."""

    SEARCH_ROM = 0xF0
    READ_ROM = 0x33
    MATCH_ROM = 0x55
    SKIP_ROM = 0xCC
    ALARM_SEARCH = 0xEC
    OVERDRIVE_SKIP = 0x3C
    OVERDRIVE_MATCH = 0x69


ROM_COMMAND_NAMES = {
    0xF0: "Search ROM",
    0x33: "Read ROM",
    0x55: "Match ROM",
    0xCC: "Skip ROM",
    0xEC: "Alarm Search",
    0x3C: "Overdrive Skip ROM",
    0x69: "Overdrive Match ROM",
}

# 1-Wire Family Codes
FAMILY_CODES = {
    0x01: "DS1990A/DS2401 Silicon Serial Number",
    0x10: "DS18S20/DS1820 Temperature Sensor",
    0x14: "DS2430A 1kb EEPROM",
    0x22: "DS1822 Econo Temperature Sensor",
    0x23: "DS2433 4kb EEPROM",
    0x28: "DS18B20 Temperature Sensor",
    0x29: "DS2408 8-Channel Addressable Switch",
    0x2D: "DS2431 1kb EEPROM",
    0x37: "DS1977 Password-Protected 32kb EEPROM",
    0x3B: "DS1825 Temperature Sensor",
    0x42: "DS28EA00 Temperature Sensor",
}


@dataclass
class OneWireTimings:
    """1-Wire protocol timing specifications in microseconds."""

    # Standard speed timings
    reset_min: float = 480.0
    reset_max: float = 960.0
    presence_min: float = 60.0
    presence_max: float = 240.0
    slot_min: float = 60.0
    slot_max: float = 120.0
    write_0_low_min: float = 60.0
    write_0_low_max: float = 120.0
    write_1_low_min: float = 1.0
    write_1_low_max: float = 15.0
    read_sample_time: float = 15.0

    @classmethod
    def overdrive(cls) -> OneWireTimings:
        """Return overdrive mode timings (approximately 10x faster)."""
        return cls(
            reset_min=48.0,
            reset_max=80.0,
            presence_min=8.0,
            presence_max=24.0,
            slot_min=6.0,
            slot_max=16.0,
            write_0_low_min=6.0,
            write_0_low_max=16.0,
            write_1_low_min=1.0,
            write_1_low_max=2.0,
            read_sample_time=1.0,
        )


@dataclass
class OneWireROMID:
    """1-Wire 64-bit ROM ID structure."""

    family_code: int  # 8 bits
    serial_number: bytes  # 48 bits (6 bytes)
    crc: int  # 8 bits

    @classmethod
    def from_bytes(cls, data: bytes) -> OneWireROMID:
        """Parse ROM ID from 8 bytes (LSB first)."""
        if len(data) < 8:
            raise ValueError("ROM ID requires 8 bytes")

        family_code = data[0]
        serial_number = data[1:7]
        crc = data[7]

        return cls(family_code=family_code, serial_number=serial_number, crc=crc)

    @property
    def family_name(self) -> str:
        """Get human-readable family name."""
        return FAMILY_CODES.get(self.family_code, f"Unknown (0x{self.family_code:02X})")

    def to_hex(self) -> str:
        """Return ROM ID as hex string."""
        return f"{self.family_code:02X}-{self.serial_number.hex().upper()}-{self.crc:02X}"

    def verify_crc(self) -> bool:
        """Verify CRC-8 of ROM ID."""
        data = bytes([self.family_code]) + self.serial_number
        return _crc8_maxim(data) == self.crc


def _crc8_maxim(data: bytes) -> int:
    """Calculate CRC-8/MAXIM (Dallas 1-Wire CRC).

    Polynomial: x^8 + x^5 + x^4 + 1 (0x31 reflected = 0x8C)

    Args:
        data: Input bytes to calculate CRC over

    Returns:
        8-bit CRC value
    """
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc


class OneWireDecoder(AsyncDecoder):
    """Dallas/Maxim 1-Wire protocol decoder.

    Decodes 1-Wire bus communication including reset/presence,
    ROM commands, and data transfers. Supports both standard
    and overdrive speeds.

    Attributes:
        id: "onewire"
        name: "1-Wire"
        channels: [data] (required)

    Example:
        >>> decoder = OneWireDecoder(mode="standard")
        >>> for packet in decoder.decode(trace):
        ...     if packet.annotations.get('rom_id'):
        ...         print(f"Device: {packet.annotations['rom_id']}")
    """

    id = "onewire"
    name = "1-Wire"
    longname = "Dallas/Maxim 1-Wire Protocol"
    desc = "1-Wire bus decoder with ROM ID extraction"

    channels = [  # noqa: RUF012
        ChannelDef("data", "DQ", "1-Wire data line", required=True),
    ]

    optional_channels = []  # noqa: RUF012

    options = [  # noqa: RUF012
        OptionDef(
            "mode",
            "Speed mode",
            "Standard or overdrive",
            default="standard",
            values=["standard", "overdrive"],
        ),
        OptionDef(
            "threshold",
            "Voltage threshold",
            "Logic threshold in volts (auto for midpoint)",
            default="auto",
            values=None,
        ),
    ]

    annotations = [  # noqa: RUF012
        ("reset", "Reset pulse"),
        ("presence", "Presence pulse"),
        ("bit", "Data bit"),
        ("byte", "Decoded byte"),
        ("rom_cmd", "ROM command"),
        ("rom_id", "ROM ID"),
        ("error", "Protocol error"),
    ]

    def __init__(
        self,
        mode: str = "standard",
        threshold: str | float = "auto",
    ) -> None:
        """Initialize 1-Wire decoder.

        Args:
            mode: Speed mode ("standard" or "overdrive").
            threshold: Logic threshold voltage or "auto".
        """
        super().__init__(mode=mode, threshold=threshold)
        self._mode = OneWireMode(mode)
        self._threshold = threshold
        self._timings = (
            OneWireTimings() if self._mode == OneWireMode.STANDARD else OneWireTimings.overdrive()
        )

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode 1-Wire protocol data.

        Args:
            trace: Input trace (digital or analog).
            **channels: Additional channel data.

        Yields:
            Decoded data as ProtocolPacket objects.

        Example:
            >>> decoder = OneWireDecoder()
            >>> for packet in decoder.decode(trace):
            ...     print(f"Command: {packet.annotations.get('rom_command')}")
        """
        # Convert to digital if needed
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            threshold = self._threshold if self._threshold != "auto" else "auto"
            digital_trace = to_digital(trace, threshold=threshold)  # type: ignore[arg-type]
        else:
            digital_trace = trace

        data = digital_trace.data.astype(bool)
        sample_rate = digital_trace.metadata.sample_rate

        # Convert timing specs to samples
        us_to_samples = sample_rate / 1_000_000

        # Find all falling and rising edges
        falling = np.where((data[:-1]) & (~data[1:]))[0]
        rising = np.where((~data[:-1]) & (data[1:]))[0]

        if len(falling) == 0:
            return

        # State machine for decoding
        decoded_bytes: list[int] = []
        current_bits: list[int] = []
        rom_id: OneWireROMID | None = None
        rom_command: int | None = None
        errors: list[str] = []
        transaction_start: float = falling[0] / sample_rate

        i = 0
        while i < len(falling):
            fall_idx = falling[i]
            fall_time = fall_idx / sample_rate

            # Find corresponding rising edge
            rise_candidates = rising[rising > fall_idx]
            if len(rise_candidates) == 0:
                break

            rise_idx = rise_candidates[0]
            low_duration_us = (rise_idx - fall_idx) / us_to_samples

            # Check for reset pulse
            if low_duration_us >= self._timings.reset_min * 0.8:
                # This is a reset pulse
                if decoded_bytes:
                    # Yield previous transaction
                    annotations = self._build_annotations(decoded_bytes, rom_command, rom_id)
                    yield ProtocolPacket(
                        timestamp=transaction_start,
                        protocol="1-wire",
                        data=bytes(decoded_bytes),
                        annotations=annotations,
                        errors=errors.copy() if errors else None,  # type: ignore[arg-type]
                    )

                # Start new transaction
                transaction_start = fall_time
                decoded_bytes = []
                current_bits = []
                rom_id = None
                rom_command = None
                errors = []

                self.put_annotation(
                    fall_time,
                    rise_idx / sample_rate,
                    AnnotationLevel.BITS,
                    "Reset",
                )

                # Look for presence pulse (pulled low by slave)
                next_falls = falling[falling > rise_idx]
                if len(next_falls) > 0:
                    next_fall = next_falls[0]
                    wait_time_us = (next_fall - rise_idx) / us_to_samples
                    if wait_time_us < self._timings.presence_max * 2:
                        # Found presence response
                        next_rises = rising[rising > next_fall]
                        if len(next_rises) > 0:
                            presence_end = next_rises[0]
                            presence_us = (presence_end - next_fall) / us_to_samples
                            if (
                                self._timings.presence_min * 0.5
                                <= presence_us
                                <= self._timings.presence_max * 1.5
                            ):
                                self.put_annotation(
                                    next_fall / sample_rate,
                                    presence_end / sample_rate,
                                    AnnotationLevel.BITS,
                                    "Presence",
                                )
                i += 1
                continue

            # Data bit - determine if 0 or 1
            # Short low pulse = write 1 or read 1
            # Long low pulse = write 0 or read 0
            if low_duration_us < self._timings.write_1_low_max * 2:
                bit = 1
            elif low_duration_us >= self._timings.write_0_low_min * 0.5:
                bit = 0
            else:
                # Ambiguous timing
                bit = 1 if low_duration_us < self._timings.slot_min * 0.5 else 0

            current_bits.append(bit)

            # Assemble bytes (LSB first)
            if len(current_bits) == 8:
                byte_val = sum(b << i for i, b in enumerate(current_bits))
                decoded_bytes.append(byte_val)
                current_bits = []

                # Check for ROM command (first byte after reset)
                if len(decoded_bytes) == 1:
                    rom_command = byte_val
                    cmd_name = ROM_COMMAND_NAMES.get(byte_val, f"Unknown (0x{byte_val:02X})")
                    self.put_annotation(
                        fall_time,
                        rise_idx / sample_rate,
                        AnnotationLevel.BYTES,
                        f"ROM Cmd: {cmd_name}",
                    )

                # Check for ROM ID (8 bytes after ROM command)
                if len(decoded_bytes) == 9 and rom_command in (
                    OneWireROMCommand.READ_ROM.value,
                    OneWireROMCommand.MATCH_ROM.value,
                ):
                    try:
                        rom_id = OneWireROMID.from_bytes(bytes(decoded_bytes[1:9]))
                        if not rom_id.verify_crc():
                            errors.append("ROM ID CRC error")
                        self.put_annotation(
                            transaction_start,
                            rise_idx / sample_rate,
                            AnnotationLevel.BYTES,
                            f"ROM: {rom_id.to_hex()}",
                        )
                    except ValueError as e:
                        errors.append(f"ROM parse error: {e}")

            i += 1

        # Yield final transaction if any
        if decoded_bytes:
            annotations = self._build_annotations(decoded_bytes, rom_command, rom_id)
            yield ProtocolPacket(
                timestamp=transaction_start,
                protocol="1-wire",
                data=bytes(decoded_bytes),
                annotations=annotations,
                errors=errors if errors else None,  # type: ignore[arg-type]
            )

    def _build_annotations(
        self,
        decoded_bytes: list[int],
        rom_command: int | None,
        rom_id: OneWireROMID | None,
    ) -> dict:  # type: ignore[type-arg]
        """Build annotation dictionary for packet."""
        annotations: dict = {  # type: ignore[type-arg]
            "mode": self._mode.value,
            "byte_count": len(decoded_bytes),
        }

        if rom_command is not None:
            annotations["rom_command"] = ROM_COMMAND_NAMES.get(rom_command, f"0x{rom_command:02X}")
            annotations["rom_command_code"] = rom_command

        if rom_id is not None:
            annotations["rom_id"] = rom_id.to_hex()
            annotations["family_code"] = rom_id.family_code
            annotations["family_name"] = rom_id.family_name
            annotations["serial_number"] = rom_id.serial_number.hex().upper()
            annotations["crc_valid"] = rom_id.verify_crc()

        return annotations


def decode_onewire(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    mode: str = "standard",
) -> list[ProtocolPacket]:
    """Convenience function to decode 1-Wire protocol data.

    Args:
        data: 1-Wire signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        mode: Speed mode ("standard" or "overdrive").

    Returns:
        List of decoded packets.

    Example:
        >>> packets = decode_onewire(signal, sample_rate=1e6)
        >>> for pkt in packets:
        ...     if 'rom_id' in pkt.annotations:
        ...         print(f"Device: {pkt.annotations['rom_id']}")
    """
    decoder = OneWireDecoder(mode=mode)
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        trace = DigitalTrace(
            data=data.astype(bool),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        return list(decoder.decode(trace))


__all__ = [
    "FAMILY_CODES",
    "ROM_COMMAND_NAMES",
    "OneWireDecoder",
    "OneWireMode",
    "OneWireROMCommand",
    "OneWireROMID",
    "OneWireTimings",
    "decode_onewire",
]
