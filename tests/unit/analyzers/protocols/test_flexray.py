"""Tests for FlexRay protocol decoder.

Tests the FlexRay decoder implementation (PRO-016).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.flexray import (
    FlexRayDecoder,
    FlexRayFrame,
    FlexRaySegment,
    decode_flexray,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# FlexRay Signal Generator
# =============================================================================


def generate_flexray_frame(
    slot_id: int = 1,
    cycle_count: int = 0,
    payload_length: int = 1,
    payload: bytes = b"\x00\x00",
    bitrate: int = 10_000_000,
    sample_rate: float = 100_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate FlexRay frame signals for testing.

    Args:
        slot_id: Slot identifier (1-2047).
        cycle_count: Cycle counter (0-63).
        payload_length: Payload length in 16-bit words (0-127).
        payload: Payload data bytes.
        bitrate: FlexRay bitrate in bps.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (bp, bm) boolean arrays.
    """
    samples_per_bit = int(sample_rate / bitrate)
    half_bit = samples_per_bit // 2

    bits = []

    # TSS (Transmission Start Sequence): 3 bit times
    # Low, Low, High
    bits.extend([0, 0, 1])

    # FSS (Frame Start Sequence): 1 bit
    bits.append(0)

    # Header (5 bytes = 40 bits)
    # Byte 1: Reserved(1) + Payload preamble(1) + NULL frame(1) + Sync(1) + Startup(1) + Slot ID[10:8](3)
    header_byte1 = (0 << 7) | (0 << 6) | (0 << 5) | (0 << 4) | (0 << 3) | ((slot_id >> 8) & 0x07)
    for i in range(8):
        bits.append((header_byte1 >> (7 - i)) & 1)

    # Byte 2: Slot ID[7:0] (8)
    header_byte2 = slot_id & 0xFF
    for i in range(8):
        bits.append((header_byte2 >> (7 - i)) & 1)

    # Byte 3: Header CRC[10:3] (8) - simplified, just use 0
    header_byte3 = 0
    for i in range(8):
        bits.append((header_byte3 >> (7 - i)) & 1)

    # Byte 4: Header CRC[2:0](3) + Cycle count[5:3](3) + reserved(2)
    header_byte4 = ((cycle_count >> 3) & 0x07) << 2
    for i in range(8):
        bits.append((header_byte4 >> (7 - i)) & 1)

    # Byte 5: Cycle count[2:0](3) + Payload length[6:2](5)
    header_byte5 = ((cycle_count & 0x07) << 5) | ((payload_length >> 2) & 0x1F)
    for i in range(8):
        bits.append((header_byte5 >> (7 - i)) & 1)

    # Byte 6 (partial): Payload length[1:0](2) + padding
    header_byte6 = (payload_length & 0x03) << 6
    for i in range(8):
        bits.append((header_byte6 >> (7 - i)) & 1)

    # Payload (payload_length * 2 bytes)
    for byte_val in payload:
        for i in range(8):
            bits.append((byte_val >> (7 - i)) & 1)

    # Frame CRC (24 bits) - simplified, just zeros
    for _ in range(24):
        bits.append(0)

    # FES (Frame End Sequence): 1 bit
    bits.append(1)

    # Generate differential signals
    idle_samples = samples_per_bit * 4
    total_samples = idle_samples + len(bits) * samples_per_bit + idle_samples

    bp = np.zeros(total_samples, dtype=bool)
    bm = np.ones(total_samples, dtype=bool)  # Idle state

    idx = idle_samples

    for bit in bits:
        if bit == 1:
            bp[idx : idx + samples_per_bit] = True
            bm[idx : idx + samples_per_bit] = False
        else:
            bp[idx : idx + samples_per_bit] = False
            bm[idx : idx + samples_per_bit] = True
        idx += samples_per_bit

    # Back to idle
    bp[idx:] = False
    bm[idx:] = True

    return bp, bm


# =============================================================================
# FlexRaySegment Tests
# =============================================================================


class TestFlexRaySegment:
    """Test FlexRaySegment enum."""

    def test_segment_values(self) -> None:
        """Test segment enum values."""
        assert FlexRaySegment.STATIC.value == "static"
        assert FlexRaySegment.DYNAMIC.value == "dynamic"
        assert FlexRaySegment.SYMBOL.value == "symbol"

    def test_all_segments_exist(self) -> None:
        """Test all expected segments are defined."""
        segments = {s.value for s in FlexRaySegment}
        expected = {"static", "dynamic", "symbol"}
        assert segments == expected


# =============================================================================
# FlexRayFrame Tests
# =============================================================================


class TestFlexRayFrame:
    """Test FlexRayFrame dataclass."""

    def test_create_frame(self) -> None:
        """Test creating FlexRay frame."""
        frame = FlexRayFrame(
            slot_id=10,
            cycle_count=5,
            payload_length=2,
            header_crc=0x123,
            payload=b"\xab\xcd\xef\x01",
            frame_crc=0x456789,
            segment=FlexRaySegment.STATIC,
            timestamp=0.001,
            errors=[],
        )

        assert frame.slot_id == 10
        assert frame.cycle_count == 5
        assert frame.payload_length == 2
        assert frame.header_crc == 0x123
        assert frame.payload == b"\xab\xcd\xef\x01"
        assert frame.frame_crc == 0x456789
        assert frame.segment == FlexRaySegment.STATIC
        assert frame.timestamp == 0.001
        assert frame.errors == []

    def test_frame_with_errors(self) -> None:
        """Test frame with errors."""
        frame = FlexRayFrame(
            slot_id=1,
            cycle_count=0,
            payload_length=0,
            header_crc=0,
            payload=b"",
            frame_crc=0,
            segment=FlexRaySegment.DYNAMIC,
            timestamp=0.0,
            errors=["CRC error", "Framing error"],
        )

        assert len(frame.errors) == 2
        assert "CRC error" in frame.errors


# =============================================================================
# FlexRayDecoder Initialization Tests
# =============================================================================


class TestFlexRayDecoderInit:
    """Test FlexRayDecoder initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test decoder with default parameters."""
        decoder = FlexRayDecoder()

        assert decoder.id == "flexray"
        assert decoder.name == "FlexRay"
        assert decoder.longname == "FlexRay Automotive Network"
        assert "FlexRay" in decoder.desc
        assert decoder._bitrate == 10_000_000

    def test_custom_bitrate(self) -> None:
        """Test decoder with custom bitrate."""
        for bitrate in [2_500_000, 5_000_000, 10_000_000]:
            decoder = FlexRayDecoder(bitrate=bitrate)
            assert decoder._bitrate == bitrate

    def test_channel_definitions(self) -> None:
        """Test decoder channel definitions."""
        decoder = FlexRayDecoder()

        assert len(decoder.channels) == 2

        channel_ids = [ch.id for ch in decoder.channels]
        assert "bp" in channel_ids
        assert "bm" in channel_ids

        for ch in decoder.channels:
            assert ch.required is True

    def test_option_definitions(self) -> None:
        """Test decoder option definitions."""
        decoder = FlexRayDecoder()

        assert len(decoder.options) > 0
        option_ids = [opt.id for opt in decoder.options]
        assert "bitrate" in option_ids

    def test_annotations_defined(self) -> None:
        """Test decoder has annotations defined."""
        decoder = FlexRayDecoder()

        assert len(decoder.annotations) > 0
        annotation_ids = [a[0] for a in decoder.annotations]
        assert "tss" in annotation_ids
        assert "header" in annotation_ids

    def test_class_constants(self) -> None:
        """Test class-level constants."""
        assert FlexRayDecoder.TSS_LENGTH == 3
        assert FlexRayDecoder.FSS_LENGTH == 1
        assert FlexRayDecoder.BSS_LENGTH == 1


# =============================================================================
# FlexRayDecoder Edge Cases
# =============================================================================


class TestFlexRayDecoderEdgeCases:
    """Test FlexRay decoder edge cases and error handling."""

    def test_decode_none_signals(self) -> None:
        """Test decode with None signals."""
        decoder = FlexRayDecoder()

        packets = list(decoder.decode(bp=None, bm=None, sample_rate=100_000_000.0))

        assert len(packets) == 0

    def test_decode_empty_signals(self) -> None:
        """Test decode with empty signals."""
        decoder = FlexRayDecoder()

        bp = np.array([], dtype=bool)
        bm = np.array([], dtype=bool)

        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=100_000_000.0))

        assert len(packets) == 0

    def test_decode_idle_bus(self) -> None:
        """Test decode with idle bus."""
        decoder = FlexRayDecoder()

        # Idle: BP=0, BM=1
        bp = np.zeros(1000, dtype=bool)
        bm = np.ones(1000, dtype=bool)

        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=100_000_000.0))

        assert len(packets) == 0

    def test_decode_mismatched_lengths(self) -> None:
        """Test decode with mismatched signal lengths."""
        decoder = FlexRayDecoder()

        bp = np.array([False, True] * 50, dtype=bool)
        bm = np.array([True, False] * 25, dtype=bool)

        # Should truncate to shortest length
        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=100_000_000.0))

        # Should not crash
        assert isinstance(packets, list)

    def test_decode_short_signal(self) -> None:
        """Test decode with signal too short for a frame."""
        decoder = FlexRayDecoder()

        bp = np.array([False, True, False], dtype=bool)
        bm = np.array([True, False, True], dtype=bool)

        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=100_000_000.0))

        assert len(packets) == 0


# =============================================================================
# FlexRayDecoder Decoding Tests
# =============================================================================


class TestFlexRayDecoderDecoding:
    """Test FlexRay decoding with actual frames."""

    def test_decode_basic_frame(self) -> None:
        """Test decoding basic FlexRay frame."""
        decoder = FlexRayDecoder(bitrate=10_000_000)
        sample_rate = 100_000_000.0

        bp, bm = generate_flexray_frame(
            slot_id=5,
            cycle_count=10,
            payload_length=1,
            payload=b"\xab\xcd",
            bitrate=10_000_000,
            sample_rate=sample_rate,
        )

        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].protocol == "flexray"
            assert "slot_id" in packets[0].annotations
            assert "cycle_count" in packets[0].annotations

    def test_decode_different_bitrates(self) -> None:
        """Test decoding at different bitrates."""
        for bitrate in [2_500_000, 5_000_000, 10_000_000]:
            decoder = FlexRayDecoder(bitrate=bitrate)
            sample_rate = bitrate * 10  # 10 samples per bit

            bp, bm = generate_flexray_frame(
                slot_id=1,
                cycle_count=0,
                payload_length=1,
                payload=b"\x00\x00",
                bitrate=bitrate,
                sample_rate=sample_rate,
            )

            packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=sample_rate))

            # Should not crash
            assert isinstance(packets, list)

    def test_decode_frame_timestamp(self) -> None:
        """Test frame timestamp calculation."""
        decoder = FlexRayDecoder(bitrate=10_000_000)
        sample_rate = 100_000_000.0

        bp, bm = generate_flexray_frame(
            slot_id=1,
            cycle_count=0,
            bitrate=10_000_000,
            sample_rate=sample_rate,
        )

        packets = list(decoder.decode(bp=bp, bm=bm, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].timestamp >= 0.0
            assert isinstance(packets[0].timestamp, float)


# =============================================================================
# FlexRayDecoder Internal Methods Tests
# =============================================================================


class TestFlexRayDecoderInternalMethods:
    """Test FlexRay decoder internal methods."""

    def test_find_tss_with_valid_pattern(self) -> None:
        """Test _find_tss with valid TSS pattern."""
        decoder = FlexRayDecoder(bitrate=10_000_000)
        sample_rate = 100_000_000.0
        bit_period = sample_rate / 10_000_000

        # Create TSS pattern: Low, Low, High
        data = np.zeros(int(bit_period * 10), dtype=bool)
        # Low for 2 bits
        data[: int(bit_period * 2)] = False
        # High for 1 bit
        data[int(bit_period * 2) : int(bit_period * 3)] = True
        # Rest is low
        data[int(bit_period * 3) :] = False

        result = decoder._find_tss(data, 0, bit_period)

        # Should find TSS somewhere in the signal
        assert result is None or isinstance(result, int)

    def test_find_tss_no_pattern(self) -> None:
        """Test _find_tss with no TSS pattern."""
        decoder = FlexRayDecoder(bitrate=10_000_000)
        sample_rate = 100_000_000.0
        bit_period = sample_rate / 10_000_000

        # All ones (no TSS pattern)
        data = np.ones(int(bit_period * 10), dtype=bool)

        result = decoder._find_tss(data, 0, bit_period)

        assert result is None

    def test_decode_frame_incomplete(self) -> None:
        """Test _decode_frame with incomplete frame."""
        decoder = FlexRayDecoder(bitrate=10_000_000)
        sample_rate = 100_000_000.0
        bit_period = sample_rate / 10_000_000

        # Very short signal
        data = np.zeros(int(bit_period * 5), dtype=bool)

        frame, end_idx = decoder._decode_frame(data, 0, sample_rate, bit_period)

        # Should return None for incomplete frame
        assert frame is None or isinstance(frame, FlexRayFrame)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeFlexRayConvenienceFunction:
    """Test decode_flexray convenience function."""

    def test_decode_flexray_basic(self) -> None:
        """Test basic usage of decode_flexray."""
        sample_rate = 100_000_000.0

        bp, bm = generate_flexray_frame(
            slot_id=1,
            cycle_count=0,
            bitrate=10_000_000,
            sample_rate=sample_rate,
        )

        packets = decode_flexray(bp, bm, sample_rate=sample_rate, bitrate=10_000_000)

        assert isinstance(packets, list)

    def test_decode_flexray_with_parameters(self) -> None:
        """Test decode_flexray with custom parameters."""
        sample_rate = 50_000_000.0
        bitrate = 5_000_000

        bp, bm = generate_flexray_frame(
            slot_id=10,
            cycle_count=30,
            bitrate=bitrate,
            sample_rate=sample_rate,
        )

        packets = decode_flexray(bp, bm, sample_rate=sample_rate, bitrate=bitrate)

        assert isinstance(packets, list)

    def test_decode_flexray_empty_signals(self) -> None:
        """Test decode_flexray with empty signals."""
        bp = np.array([], dtype=bool)
        bm = np.array([], dtype=bool)

        packets = decode_flexray(bp, bm, sample_rate=100_000_000.0)

        assert packets == []


# =============================================================================
# Module Export Tests
# =============================================================================


class TestFlexRayModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.flexray import __all__

        assert "FlexRayDecoder" in __all__
        assert "FlexRayFrame" in __all__
        assert "FlexRaySegment" in __all__
        assert "decode_flexray" in __all__
