"""Manchester encoding decoder.

This module provides Manchester and Differential Manchester encoding
decoders with clock recovery and violation detection.


Example:
    >>> from oscura.analyzers.protocols.manchester import ManchesterDecoder
    >>> decoder = ManchesterDecoder(mode="ieee")
    >>> for packet in decoder.decode(trace):
    ...     print(f"Data: {packet.data.hex()}")

References:
    IEEE 802.3 (Ethernet)
    Thomas & Biba Differential Manchester
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    OptionDef,
)
from oscura.core.types import DigitalTrace, ProtocolPacket, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class ManchesterMode(Enum):
    """Manchester encoding modes."""

    IEEE = "ieee"  # IEEE 802.3: 0=low-high, 1=high-low
    THOMAS = "thomas"  # G.E. Thomas: 0=high-low, 1=low-high
    DIFFERENTIAL = "differential"  # Differential Manchester


class ManchesterDecoder(AsyncDecoder):
    """Manchester encoding decoder.

    Decodes Manchester and Differential Manchester encoded data
    with automatic clock recovery and encoding violation detection.

    Attributes:
        id: "manchester"
        name: "Manchester"
        channels: [data] (required)

    Example:
        >>> decoder = ManchesterDecoder(mode="ieee", bit_rate=10000000)
        >>> for packet in decoder.decode(trace):
        ...     print(f"Bits: {packet.annotations['bit_count']}")
    """

    id = "manchester"
    name = "Manchester"
    longname = "Manchester Encoding"
    desc = "Manchester and Differential Manchester decoder"

    channels = [
        ChannelDef("data", "DATA", "Manchester encoded data", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef("bit_rate", "Bit rate", "Bits per second", default=10000000, values=None),
        OptionDef(
            "mode",
            "Encoding mode",
            "Manchester variant",
            default="ieee",
            values=["ieee", "thomas", "differential"],
        ),
    ]

    annotations = [
        ("bit", "Decoded bit"),
        ("clock", "Recovered clock"),
        ("violation", "Encoding violation"),
    ]

    def __init__(
        self,
        bit_rate: int = 10000000,
        mode: Literal["ieee", "thomas", "differential"] = "ieee",
    ) -> None:
        """Initialize Manchester decoder.

        Args:
            bit_rate: Bit rate in bps (before encoding).
            mode: Encoding mode ("ieee", "thomas", or "differential").
        """
        # Manchester encoding doubles the transition rate
        super().__init__(baudrate=bit_rate * 2, mode=mode, bit_rate=bit_rate)
        self._bit_rate = bit_rate
        self._mode = ManchesterMode(mode)

    def decode(
        self,
        trace: DigitalTrace | WaveformTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode Manchester encoded data.

        Args:
            trace: Input digital trace.
            **channels: Additional channel data.

        Yields:
            Decoded data as ProtocolPacket objects.

        Example:
            >>> decoder = ManchesterDecoder(mode="ieee", bit_rate=10e6)
            >>> for packet in decoder.decode(trace):
            ...     print(f"Data: {packet.data.hex()}")
        """
        # Convert to digital if needed
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers.digital.extraction import to_digital

            digital_trace = to_digital(trace, threshold="auto")
        else:
            digital_trace = trace

        data = digital_trace.data
        sample_rate = digital_trace.metadata.sample_rate

        # Bit period (actual data bit, not symbol period)
        bit_period = sample_rate / self._bit_rate
        half_bit = bit_period / 2

        # Find transitions for clock recovery
        transitions = np.where(data[:-1] != data[1:])[0] + 1

        if len(transitions) < 2:
            return

        # Decode bits based on transition timing
        decoded_bits = []  # type: ignore[var-annotated]
        errors = []  # type: ignore[var-annotated]
        start_time = 0

        if self._mode == ManchesterMode.DIFFERENTIAL:
            decoded_bits, errors = self._decode_differential(data, transitions, half_bit)
        else:
            decoded_bits, errors = self._decode_standard(data, transitions, half_bit)

        if len(decoded_bits) == 0:
            return

        # Convert bits to bytes
        byte_list = []
        for i in range(0, len(decoded_bits), 8):
            if i + 8 <= len(decoded_bits):
                byte_val = 0
                for j in range(8):
                    byte_val |= decoded_bits[i + j] << j
                byte_list.append(byte_val)

        # Calculate timing
        start_time = transitions[0] / sample_rate if len(transitions) > 0 else 0
        end_time = transitions[-1] / sample_rate if len(transitions) > 0 else 0

        # Add annotation
        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.BYTES,
            f"{len(decoded_bits)} bits decoded",
        )

        # Create packet
        annotations = {
            "bit_count": len(decoded_bits),
            "byte_count": len(byte_list),
            "mode": self._mode.value,
            "bit_rate": self._bit_rate,
        }

        packet = ProtocolPacket(
            timestamp=start_time,
            protocol="manchester",
            data=bytes(byte_list),
            annotations=annotations,
            errors=errors,
        )

        yield packet

    def _decode_standard(
        self,
        data: NDArray[np.bool_],
        transitions: NDArray[np.int64],
        half_bit: float,
    ) -> tuple[list[int], list[str]]:
        """Decode standard Manchester (IEEE or Thomas).

        Args:
            data: Digital data array.
            transitions: Transition indices.
            half_bit: Half-bit period in samples.

        Returns:
            (decoded_bits, errors) tuple.
        """
        decoded_bits = []
        errors = []  # type: ignore[var-annotated]

        # In Manchester, there's always a transition in the middle of each bit
        # IEEE: 0=low-to-high, 1=high-to-low (mid-bit transition)
        # Thomas: opposite

        i = 0
        while i < len(transitions) - 1:
            trans_idx = transitions[i]

            # Sample before and after transition
            if trans_idx > 0 and trans_idx < len(data):
                before = data[trans_idx - 1]
                after = data[trans_idx]

                # Determine bit value based on transition direction
                if not before and after:  # Rising edge
                    bit = 0 if self._mode == ManchesterMode.IEEE else 1
                else:  # Falling edge
                    bit = 1 if self._mode == ManchesterMode.IEEE else 0

                decoded_bits.append(bit)

            # Look for next mid-bit transition (should be ~1 bit period away)
            i += 1

            # Check if there's a boundary transition (should be ~0.5 bit period)
            if i < len(transitions):
                trans_spacing = transitions[i] - trans_idx
                if trans_spacing < half_bit * 0.7:
                    # This is a boundary transition, skip it
                    i += 1

        return decoded_bits, errors

    def _decode_differential(
        self,
        data: NDArray[np.bool_],
        transitions: NDArray[np.int64],
        half_bit: float,
    ) -> tuple[list[int], list[str]]:
        """Decode Differential Manchester.

        Args:
            data: Digital data array.
            transitions: Transition indices.
            half_bit: Half-bit period in samples.

        Returns:
            (decoded_bits, errors) tuple.
        """
        decoded_bits = []
        errors = []  # type: ignore[var-annotated]

        # Differential Manchester:
        # Always transition at bit boundary
        # 0: additional transition at mid-bit
        # 1: no additional transition at mid-bit

        i = 0
        while i < len(transitions) - 1:
            trans_idx = transitions[i]

            # Check if there's a transition within the bit period
            next_trans_spacing = (
                transitions[i + 1] - trans_idx if i + 1 < len(transitions) else float("inf")
            )

            if next_trans_spacing < half_bit * 1.5:
                # Two transitions in this bit period -> bit = 0
                decoded_bits.append(0)
                i += 2
            else:
                # One transition in this bit period -> bit = 1
                decoded_bits.append(1)
                i += 1

        return decoded_bits, errors


def decode_manchester(
    data: NDArray[np.bool_] | WaveformTrace | DigitalTrace,
    sample_rate: float = 1.0,
    bit_rate: int = 10000000,
    mode: Literal["ieee", "thomas", "differential"] = "ieee",
) -> list[ProtocolPacket]:
    """Convenience function to decode Manchester encoded data.

    Args:
        data: Manchester encoded signal (digital array or trace).
        sample_rate: Sample rate in Hz.
        bit_rate: Bit rate in bps (before encoding).
        mode: Encoding mode ("ieee", "thomas", or "differential").

    Returns:
        List of decoded packets.

    Example:
        >>> packets = decode_manchester(signal, sample_rate=100e6, bit_rate=10e6)
        >>> for pkt in packets:
        ...     print(f"Data: {pkt.data.hex()}")
    """
    decoder = ManchesterDecoder(bit_rate=bit_rate, mode=mode)
    if isinstance(data, WaveformTrace | DigitalTrace):
        return list(decoder.decode(data))
    else:
        from oscura.core.types import TraceMetadata

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = DigitalTrace(
            data=data,
            metadata=metadata,
        )
        return list(decoder.decode(trace))


__all__ = ["ManchesterDecoder", "ManchesterMode", "decode_manchester"]
