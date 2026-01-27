"""SPI protocol decoder.

This module provides SPI (Serial Peripheral Interface) protocol
decoding with configurable CPOL/CPHA modes and word sizes.


Example:
    >>> from oscura.analyzers.protocols.spi import SPIDecoder
    >>> decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
    >>> for packet in decoder.decode(clk=clock, mosi=mosi, miso=miso, cs=cs):
    ...     print(f"TX: {packet.annotations['mosi'].hex()}")

References:
    SPI Specification (Motorola)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    ChannelDef,
    OptionDef,
    SyncDecoder,
)
from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class SPIDecoder(SyncDecoder):
    """SPI protocol decoder.

    Decodes SPI bus transactions with configurable clock polarity,
    clock phase, and word size.

    Mode mapping:
        - Mode 0: CPOL=0, CPHA=0 (sample on rising, shift on falling)
        - Mode 1: CPOL=0, CPHA=1 (sample on falling, shift on rising)
        - Mode 2: CPOL=1, CPHA=0 (sample on falling, shift on rising)
        - Mode 3: CPOL=1, CPHA=1 (sample on rising, shift on falling)

    Example:
        >>> decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        >>> for packet in decoder.decode(trace, clk=clk, mosi=mosi, miso=miso):
        ...     print(f"MOSI: {packet.annotations['mosi'].hex()}")
    """

    id = "spi"
    name = "SPI"
    longname = "Serial Peripheral Interface"
    desc = "SPI bus protocol decoder"

    channels = [
        ChannelDef("clk", "CLK", "Clock signal", required=True),
        ChannelDef("mosi", "MOSI", "Master Out Slave In", required=True),
    ]

    optional_channels = [
        ChannelDef("miso", "MISO", "Master In Slave Out", required=False),
        ChannelDef("cs", "CS#", "Chip Select (active low)", required=False),
    ]

    options = [
        OptionDef("cpol", "Clock Polarity", "Clock idle state", default=0, values=[0, 1]),
        OptionDef("cpha", "Clock Phase", "Sample edge", default=0, values=[0, 1]),
        OptionDef(
            "word_size",
            "Word size",
            "Bits per word",
            default=8,
            values=[4, 8, 16, 24, 32],
        ),
        OptionDef("bit_order", "Bit order", "Bit order", default="msb", values=["msb", "lsb"]),
        OptionDef(
            "cs_polarity",
            "CS polarity",
            "Chip select polarity",
            default=0,
            values=[0, 1],
        ),
    ]

    annotations = [
        ("bit", "Bit value"),
        ("byte", "Decoded byte"),
        ("word", "Decoded word"),
        ("transfer", "Complete transfer"),
    ]

    def __init__(
        self,
        cpol: Literal[0, 1] = 0,
        cpha: Literal[0, 1] = 0,
        word_size: int = 8,
        bit_order: Literal["msb", "lsb"] = "msb",
        cs_polarity: Literal[0, 1] = 0,
    ) -> None:
        """Initialize SPI decoder.

        Args:
            cpol: Clock polarity (0=idle low, 1=idle high).
            cpha: Clock phase (0=sample on first edge, 1=sample on second edge).
            word_size: Bits per word.
            bit_order: Bit order ("msb" or "lsb").
            cs_polarity: CS active level (0=active low, 1=active high).
        """
        super().__init__(
            cpol=cpol,
            cpha=cpha,
            word_size=word_size,
            bit_order=bit_order,
            cs_polarity=cs_polarity,
        )
        self._cpol = cpol
        self._cpha = cpha
        self._word_size = word_size
        self._bit_order = bit_order
        self._cs_polarity = cs_polarity

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        clk: NDArray[np.bool_] | None = None,
        mosi: NDArray[np.bool_] | None = None,
        miso: NDArray[np.bool_] | None = None,
        cs: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode SPI transactions.

        Args:
            trace: Optional primary trace (uses clk if provided).
            clk: Clock signal.
            mosi: Master Out Slave In data.
            miso: Master In Slave Out data (optional).
            cs: Chip Select signal (optional).
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded SPI words as ProtocolPacket objects.

        Example:
            >>> decoder = SPIDecoder(cpol=0, cpha=0)
            >>> for pkt in decoder.decode(clk=clk, mosi=mosi, miso=miso, sample_rate=1e9):
            ...     print(f"Word: 0x{pkt.annotations['mosi_value']:04X}")
        """
        clk, mosi, miso, cs, sample_rate = self._prepare_spi_signals(
            trace, clk, mosi, miso, cs, sample_rate
        )

        if clk is None or mosi is None:
            return

        edges = self._find_sample_edges(clk)
        if len(edges) == 0:
            return

        mosi_bits: list[int] = []
        miso_bits: list[int] = []
        word_start_idx = edges[0]
        word_num = 0

        for edge_idx in edges:
            if not self._is_cs_active(cs, edge_idx):
                mosi_bits = []
                miso_bits = []
                continue

            mosi_bits, miso_bits = self._sample_data_lines(
                mosi, miso, edge_idx, mosi_bits, miso_bits
            )

            if len(mosi_bits) >= self._word_size:
                packet = self._build_spi_word_packet(
                    mosi_bits, miso_bits, word_start_idx, edge_idx, sample_rate, word_num
                )

                yield packet

                mosi_bits = mosi_bits[self._word_size :]
                miso_bits = miso_bits[self._word_size :] if miso_bits else []
                word_start_idx = edge_idx
                word_num += 1

    def _prepare_spi_signals(
        self,
        trace: DigitalTrace | None,
        clk: NDArray[np.bool_] | None,
        mosi: NDArray[np.bool_] | None,
        miso: NDArray[np.bool_] | None,
        cs: NDArray[np.bool_] | None,
        sample_rate: float,
    ) -> tuple[
        NDArray[np.bool_] | None,
        NDArray[np.bool_] | None,
        NDArray[np.bool_] | None,
        NDArray[np.bool_] | None,
        float,
    ]:
        """Prepare and align all SPI signals.

        Args:
            trace: Optional trace.
            clk: Clock signal.
            mosi: MOSI signal.
            miso: MISO signal.
            cs: CS signal.
            sample_rate: Sample rate.

        Returns:
            Tuple of aligned signals and sample rate.
        """
        if trace is not None:
            clk = trace.data
            sample_rate = trace.metadata.sample_rate

        if clk is None or mosi is None:
            return None, None, None, None, sample_rate

        n_samples = min(len(clk), len(mosi))
        if miso is not None:
            n_samples = min(n_samples, len(miso))
        if cs is not None:
            n_samples = min(n_samples, len(cs))

        clk = clk[:n_samples]
        mosi = mosi[:n_samples]
        if miso is not None:
            miso = miso[:n_samples]
        if cs is not None:
            cs = cs[:n_samples]

        return clk, mosi, miso, cs, sample_rate

    def _find_sample_edges(self, clk: NDArray[np.bool_]) -> NDArray[np.intp]:
        """Find clock edges for sampling based on CPOL/CPHA.

        Args:
            clk: Clock signal.

        Returns:
            Array of edge indices.
        """
        sample_edge = self._determine_sample_edge()

        if sample_edge == "rising":
            return np.where(~clk[:-1] & clk[1:])[0] + 1
        else:
            return np.where(clk[:-1] & ~clk[1:])[0] + 1

    def _determine_sample_edge(self) -> str:
        """Determine which clock edge to sample on.

        Returns:
            "rising" or "falling".
        """
        if self._cpol == 0:
            return "rising" if self._cpha == 0 else "falling"
        elif self._cpha == 0:
            return "falling"
        else:
            return "rising"

    def _is_cs_active(self, cs: NDArray[np.bool_] | None, idx: int) -> bool:
        """Check if chip select is active.

        Args:
            cs: CS signal or None.
            idx: Index to check.

        Returns:
            True if CS active or not provided.
        """
        if cs is None:
            return True
        return bool(cs[idx] == (self._cs_polarity == 1))

    def _sample_data_lines(
        self,
        mosi: NDArray[np.bool_],
        miso: NDArray[np.bool_] | None,
        idx: int,
        mosi_bits: list[int],
        miso_bits: list[int],
    ) -> tuple[list[int], list[int]]:
        """Sample MOSI and MISO lines at edge.

        Args:
            mosi: MOSI signal.
            miso: MISO signal or None.
            idx: Edge index.
            mosi_bits: Current MOSI bit buffer.
            miso_bits: Current MISO bit buffer.

        Returns:
            Updated (mosi_bits, miso_bits).
        """
        mosi_bit = 1 if mosi[idx] else 0
        mosi_bits.append(mosi_bit)

        if miso is not None:
            miso_bit = 1 if miso[idx] else 0
            miso_bits.append(miso_bit)

        return mosi_bits, miso_bits

    def _build_spi_word_packet(
        self,
        mosi_bits: list[int],
        miso_bits: list[int],
        start_idx: int,
        end_idx: int,
        sample_rate: float,
        word_num: int,
    ) -> ProtocolPacket:
        """Build SPI word packet.

        Args:
            mosi_bits: MOSI bit buffer.
            miso_bits: MISO bit buffer.
            start_idx: Word start index.
            end_idx: Word end index.
            sample_rate: Sample rate.
            word_num: Word number.

        Returns:
            Protocol packet.
        """
        mosi_value = self._bits_to_value(mosi_bits[: self._word_size])
        miso_value = self._bits_to_value(miso_bits[: self._word_size]) if miso_bits else None

        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate

        byte_count = (self._word_size + 7) // 8
        mosi_bytes = mosi_value.to_bytes(byte_count, "big")

        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.WORDS,
            f"MOSI: 0x{mosi_value:0{byte_count * 2}X}",
            data=mosi_bytes,
        )

        annotations = {
            "word_num": word_num,
            "mosi_bits": mosi_bits[: self._word_size],
            "mosi_value": mosi_value,
            "word_size": self._word_size,
            "mode": self._cpol * 2 + self._cpha,
        }

        if miso_value is not None:
            annotations["miso_bits"] = miso_bits[: self._word_size]
            annotations["miso_value"] = miso_value

        return ProtocolPacket(
            timestamp=start_time,
            protocol="spi",
            data=mosi_bytes,
            annotations=annotations,
            errors=[],
        )

    def _bits_to_value(self, bits: list[int]) -> int:
        """Convert bit list to integer value.

        Args:
            bits: List of bit values (0 or 1).

        Returns:
            Integer value.
        """
        value = 0

        if self._bit_order == "msb":
            for bit in bits:
                value = (value << 1) | bit
        else:
            for i, bit in enumerate(bits):
                value |= bit << i

        return value


def decode_spi(
    clk: NDArray[np.bool_],
    mosi: NDArray[np.bool_] | None = None,
    miso: NDArray[np.bool_] | None = None,
    cs: NDArray[np.bool_] | None = None,
    sample_rate: float = 1.0,
    cpol: Literal[0, 1] = 0,
    cpha: Literal[0, 1] = 0,
    word_size: int = 8,
    bit_order: Literal["msb", "lsb"] = "msb",
) -> list[ProtocolPacket]:
    """Convenience function to decode SPI transactions.

    Args:
        clk: Clock signal.
        mosi: Master Out Slave In signal (optional).
        miso: Master In Slave Out signal (optional).
        cs: Chip select signal (optional, active low).
        sample_rate: Sample rate in Hz.
        cpol: Clock polarity (0 or 1).
        cpha: Clock phase (0 or 1).
        word_size: Bits per word (default 8).
        bit_order: Bit order ("msb" or "lsb").

    Returns:
        List of decoded SPI transactions.

    Example:
        >>> packets = decode_spi(clk, mosi=mosi, miso=miso, sample_rate=10e6)
        >>> for pkt in packets:
        ...     print(f"MOSI: {pkt.annotations['mosi'].hex()}")
    """
    decoder = SPIDecoder(cpol=cpol, cpha=cpha, word_size=word_size, bit_order=bit_order)
    return list(decoder.decode(clk=clk, mosi=mosi, miso=miso, cs=cs, sample_rate=sample_rate))


__all__ = ["SPIDecoder", "decode_spi"]
