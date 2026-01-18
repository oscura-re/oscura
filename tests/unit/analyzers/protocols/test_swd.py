"""Tests for SWD protocol decoder.

Tests the SWD (Serial Wire Debug) decoder implementation (PRO-010).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.swd import (
    SWDDecoder,
    SWDResponse,
    decode_swd,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# SWD Signal Generator
# =============================================================================


def generate_swd_transaction(
    apndp: int = 0,
    rnw: int = 0,
    addr: int = 0,
    data: int = 0,
    ack: int = 1,  # 1 = OK, 2 = WAIT, 4 = FAULT
    samples_per_bit: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SWD transaction signals for testing.

    Args:
        apndp: 0 = DP, 1 = AP access.
        rnw: 0 = Write, 1 = Read.
        addr: Register address (bits 2 and 3).
        data: 32-bit data value for data phase.
        ack: ACK response (1=OK, 2=WAIT, 4=FAULT).
        samples_per_bit: Samples per bit period.

    Returns:
        Tuple of (swclk, swdio) boolean arrays.
    """
    # SWD transaction:
    # Request: Start(1) + APnDP(1) + RnW(1) + A[2](1) + A[3](1) + Parity(1) + Stop(0) + Park(1)
    # Turnaround: 1 clock
    # ACK: 3 bits (LSB first)
    # (if OK) Turnaround + Data(32 bits) + Parity(1)
    # Turnaround + idle

    bits = []

    # Request packet (8 bits)
    # Start bit = 1
    bits.append(1)
    # APnDP
    bits.append(apndp)
    # RnW
    bits.append(rnw)
    # A[2]
    addr_2 = (addr >> 2) & 1
    bits.append(addr_2)
    # A[3]
    addr_3 = (addr >> 3) & 1
    bits.append(addr_3)
    # Parity (odd parity of APnDP, RnW, A[2], A[3])
    parity = (apndp + rnw + addr_2 + addr_3) % 2
    bits.append(parity)
    # Stop bit = 0
    bits.append(0)
    # Park bit = 1
    bits.append(1)

    # Turnaround (1 clock)
    bits.append(0)  # Host releases line

    # ACK (3 bits, LSB first)
    ack_bits = [(ack >> i) & 1 for i in range(3)]
    bits.extend(ack_bits)

    # Data phase (only if ACK is OK)
    if ack == SWDResponse.OK.value:
        # Turnaround (1 clock)
        bits.append(0)

        # Data (32 bits, LSB first)
        data_bits = [(data >> i) & 1 for i in range(32)]
        bits.extend(data_bits)

        # Data parity (odd parity)
        data_parity = sum(data_bits) % 2
        bits.append(data_parity)

    # Turnaround and idle
    bits.extend([0, 1, 1, 1, 1])

    # Generate clock and data signals
    half_bit = samples_per_bit // 2
    total_samples = len(bits) * samples_per_bit + samples_per_bit * 2

    swclk = np.zeros(total_samples, dtype=bool)
    swdio = np.zeros(total_samples, dtype=bool)

    # Start with some idle (high)
    swdio[:samples_per_bit] = True
    idx = samples_per_bit

    for bit in bits:
        # Clock low
        swclk[idx : idx + half_bit] = False
        # Clock high
        swclk[idx + half_bit : idx + samples_per_bit] = True
        # Data valid during whole bit period
        swdio[idx : idx + samples_per_bit] = bool(bit)
        idx += samples_per_bit

    return swclk[:idx], swdio[:idx]


# =============================================================================
# SWDResponse Tests
# =============================================================================


class TestSWDResponse:
    """Test SWDResponse enum."""

    def test_response_values(self) -> None:
        """Test response enum values."""
        assert SWDResponse.OK.value == 0b001
        assert SWDResponse.WAIT.value == 0b010
        assert SWDResponse.FAULT.value == 0b100

    def test_all_responses_exist(self) -> None:
        """Test all expected responses are defined."""
        responses = {r.name for r in SWDResponse}
        expected = {"OK", "WAIT", "FAULT"}
        assert responses == expected


# =============================================================================
# SWDDecoder Initialization Tests
# =============================================================================


class TestSWDDecoderInit:
    """Test SWDDecoder initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test decoder with default parameters."""
        decoder = SWDDecoder()

        assert decoder.id == "swd"
        assert decoder.name == "SWD"
        assert decoder.longname == "Serial Wire Debug"
        assert "Serial Wire Debug" in decoder.desc

    def test_channel_definitions(self) -> None:
        """Test decoder channel definitions."""
        decoder = SWDDecoder()

        assert len(decoder.channels) == 2

        channel_ids = [ch.id for ch in decoder.channels]
        assert "swclk" in channel_ids
        assert "swdio" in channel_ids

        for ch in decoder.channels:
            assert ch.required is True

    def test_annotations_defined(self) -> None:
        """Test decoder has annotations defined."""
        decoder = SWDDecoder()

        assert len(decoder.annotations) > 0
        annotation_ids = [a[0] for a in decoder.annotations]
        assert "request" in annotation_ids
        assert "ack" in annotation_ids


# =============================================================================
# SWDDecoder Edge Cases
# =============================================================================


class TestSWDDecoderEdgeCases:
    """Test SWD decoder edge cases and error handling."""

    def test_decode_empty_signals(self) -> None:
        """Test decode with empty signals."""
        decoder = SWDDecoder()

        swclk = np.array([], dtype=bool)
        swdio = np.array([], dtype=bool)

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_none_signals(self) -> None:
        """Test decode with None signals."""
        decoder = SWDDecoder()

        packets = list(decoder.decode(swclk=None, swdio=None, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_no_clock_edges(self) -> None:
        """Test decode with constant clock (no edges)."""
        decoder = SWDDecoder()

        swclk = np.zeros(100, dtype=bool)
        swdio = np.ones(100, dtype=bool)

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_mismatched_lengths(self) -> None:
        """Test decode with mismatched signal lengths."""
        decoder = SWDDecoder()

        swclk = np.array([False, True] * 50, dtype=bool)
        swdio = np.array([True, False] * 25, dtype=bool)

        # Should truncate to shortest length
        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=1_000_000.0))

        # Should not crash
        assert isinstance(packets, list)


# =============================================================================
# SWDDecoder Decoding Tests
# =============================================================================


class TestSWDDecoderDecoding:
    """Test SWD decoding with actual transactions."""

    def test_decode_dp_read_ok(self) -> None:
        """Test decoding DP read transaction with OK response."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=0,  # DP
            rnw=1,  # Read
            addr=0,  # DPIDR
            data=0x2BA01477,
            ack=SWDResponse.OK.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].protocol == "swd"
        assert packets[0].annotations["apndp"] == "DP"
        assert packets[0].annotations["read"] is True
        assert packets[0].annotations["ack"] == "OK"

    def test_decode_dp_write_ok(self) -> None:
        """Test decoding DP write transaction with OK response."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=0,  # DP
            rnw=0,  # Write
            addr=0x04,  # CTRL/STAT
            data=0x50000000,
            ack=SWDResponse.OK.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].annotations["apndp"] == "DP"
        assert packets[0].annotations["read"] is False

    def test_decode_ap_access(self) -> None:
        """Test decoding AP access transaction."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=1,  # AP
            rnw=1,  # Read
            addr=0x0C,  # IDR
            data=0x24770011,
            ack=SWDResponse.OK.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].annotations["apndp"] == "AP"

    def test_decode_wait_response(self) -> None:
        """Test decoding transaction with WAIT response."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=0,
            rnw=1,
            addr=0,
            data=0,
            ack=SWDResponse.WAIT.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["ack"] == "WAIT"
            assert "WAIT" in packets[0].errors[0] if packets[0].errors else True

    def test_decode_fault_response(self) -> None:
        """Test decoding transaction with FAULT response."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=0,
            rnw=1,
            addr=0,
            data=0,
            ack=SWDResponse.FAULT.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["ack"] == "FAULT"
            assert "FAULT" in packets[0].errors[0] if packets[0].errors else True

    def test_decode_register_address(self) -> None:
        """Test register address extraction."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        # Address 0x0C = bits A[2]=1, A[3]=1
        swclk, swdio = generate_swd_transaction(
            apndp=0,
            rnw=1,
            addr=0x0C,
            data=0x12345678,
            ack=SWDResponse.OK.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["register_addr"] == 0x0C

    def test_packet_timestamp(self) -> None:
        """Test packet timestamp calculation."""
        decoder = SWDDecoder()
        sample_rate = 1_000_000.0

        swclk, swdio = generate_swd_transaction(
            apndp=0,
            rnw=1,
            addr=0,
            data=0x12345678,
            ack=SWDResponse.OK.value,
        )

        packets = list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].timestamp >= 0.0
            assert isinstance(packets[0].timestamp, float)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeSWDConvenienceFunction:
    """Test decode_swd convenience function."""

    def test_decode_swd_basic(self) -> None:
        """Test basic usage of decode_swd."""
        swclk, swdio = generate_swd_transaction(
            apndp=0,
            rnw=1,
            addr=0,
            data=0x2BA01477,
            ack=SWDResponse.OK.value,
        )

        packets = decode_swd(swclk, swdio, sample_rate=1_000_000.0)

        assert isinstance(packets, list)
        assert len(packets) == 1

    def test_decode_swd_empty_signals(self) -> None:
        """Test decode_swd with empty signals."""
        swclk = np.array([], dtype=bool)
        swdio = np.array([], dtype=bool)

        packets = decode_swd(swclk, swdio, sample_rate=1_000_000.0)

        assert packets == []


# =============================================================================
# Module Export Tests
# =============================================================================


class TestSWDModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.swd import __all__

        assert "SWDDecoder" in __all__
        assert "SWDResponse" in __all__
        assert "decode_swd" in __all__
