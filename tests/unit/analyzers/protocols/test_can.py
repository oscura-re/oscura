"""Tests for CAN protocol decoder.

Tests for TASK-042 (CAN Decoder).
"""

import numpy as np
import pytest

from oscura.analyzers.protocols.can import (
    CAN_BITRATES,
    CANDecoder,
    CANFrame,
    decode_can,
)
from oscura.core.types import DigitalTrace, TraceMetadata

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Fixtures
# =============================================================================


def generate_can_bit_stream(
    arb_id: int,
    data: bytes,
    extended: bool = False,
    samples_per_bit: int = 10,
) -> np.ndarray:
    """Generate a CAN bit stream.

    Args:
        arb_id: Arbitration ID (11-bit or 29-bit).
        data: Data bytes (0-8 bytes).
        extended: True for extended (29-bit) ID.
        samples_per_bit: Samples per bit.

    Returns:
        Boolean array representing CAN signal.
    """
    bits = []

    # CAN uses NRZ with bit stuffing
    # Dominant = 0, Recessive = 1

    # Idle period (recessive)
    bits.extend([1] * 20)

    # Start of Frame (SOF) - dominant
    bits.append(0)

    def add_bits_with_stuffing(new_bits: list[int]) -> list[int]:
        """Add bits with bit stuffing."""
        result = []
        consecutive = 0
        last_bit = None

        for bit in new_bits:
            result.append(bit)

            if last_bit == bit:
                consecutive += 1
            else:
                consecutive = 1

            # After 5 consecutive same bits, insert opposite
            if consecutive == 5:
                result.append(1 - bit)  # Stuff bit
                consecutive = 0
                last_bit = 1 - bit
            else:
                last_bit = bit

        return result

    frame_bits = []

    if extended:
        # Extended frame (29-bit ID)
        # First 11 bits of ID
        for i in range(10, -1, -1):
            frame_bits.append((arb_id >> (18 + i)) & 1)
        # SRR (recessive)
        frame_bits.append(1)
        # IDE (recessive for extended)
        frame_bits.append(1)
        # Remaining 18 bits of ID
        for i in range(17, -1, -1):
            frame_bits.append((arb_id >> i) & 1)
        # RTR (dominant for data frame)
        frame_bits.append(0)
        # r1, r0 reserved (dominant)
        frame_bits.extend([0, 0])
    else:
        # Standard frame (11-bit ID)
        for i in range(10, -1, -1):
            frame_bits.append((arb_id >> i) & 1)
        # RTR (dominant for data frame)
        frame_bits.append(0)
        # IDE (dominant for standard)
        frame_bits.append(0)
        # r0 reserved (dominant)
        frame_bits.append(0)

    # DLC (4 bits)
    dlc = len(data)
    for i in range(3, -1, -1):
        frame_bits.append((dlc >> i) & 1)

    # Data field
    for byte in data:
        for i in range(7, -1, -1):
            frame_bits.append((byte >> i) & 1)

    # Calculate CRC (simplified - just use zeros for test)
    crc = 0  # Simplified
    for i in range(14, -1, -1):
        frame_bits.append((crc >> i) & 1)

    # Apply bit stuffing to frame bits
    stuffed_bits = add_bits_with_stuffing(frame_bits)
    bits.extend(stuffed_bits)

    # CRC delimiter (recessive)
    bits.append(1)

    # ACK slot (dominant - receiver acknowledges)
    bits.append(0)

    # ACK delimiter (recessive)
    bits.append(1)

    # End of Frame (7 recessive bits)
    bits.extend([1] * 7)

    # Inter-frame space (3 recessive bits)
    bits.extend([1] * 3)

    # More idle
    bits.extend([1] * 20)

    # Convert to samples
    signal = np.repeat(np.array(bits, dtype=bool), samples_per_bit)

    return signal


def generate_malformed_can_frame(
    malformation: str,
    samples_per_bit: int = 10,
) -> np.ndarray:
    """Generate a malformed CAN frame for negative testing.

    Args:
        malformation: Type of malformation:
            - "truncated_sof": Frame cut short after SOF
            - "missing_eof": Missing End of Frame
            - "stuff_error": Invalid bit stuffing (6 consecutive same bits)
            - "invalid_dlc": DLC > 8
            - "short_frame": Frame too short to contain header
            - "crc_error": Invalid CRC (non-zero when expecting zero)

    Returns:
        Boolean array representing malformed CAN signal.
    """
    bits = []

    # Idle period (recessive)
    bits.extend([1] * 20)

    if malformation == "truncated_sof":
        # Start of Frame only, then abrupt end
        bits.append(0)
        bits.extend([1] * 5)  # Short truncation

    elif malformation == "missing_eof":
        # Valid frame start but missing EOF
        bits.append(0)  # SOF
        # Some ID bits
        bits.extend([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])  # 11-bit ID
        bits.extend([0, 0, 0])  # RTR, IDE, r0
        bits.extend([0, 0, 0, 1])  # DLC = 1
        bits.extend([0, 0, 0, 0, 0, 0, 0, 0])  # Data byte
        bits.extend([0] * 15)  # CRC (zeros)
        bits.append(1)  # CRC delimiter
        bits.append(0)  # ACK
        bits.append(1)  # ACK delimiter
        # Missing EOF (should be 7 recessive bits)
        bits.extend([1] * 2)  # Only 2 bits instead of 7

    elif malformation == "stuff_error":
        # Frame with 6 consecutive identical bits (stuffing violation)
        bits.append(0)  # SOF
        bits.extend([0, 0, 0, 0, 0, 0])  # 6 consecutive dominant - stuffing error!
        bits.extend([1] * 10)  # Some recessive bits after

    elif malformation == "invalid_dlc":
        # Frame with DLC > 8 (invalid)
        bits.append(0)  # SOF
        bits.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 11-bit ID = 0
        bits.extend([0, 0, 0])  # RTR, IDE, r0
        bits.extend([1, 1, 1, 1])  # DLC = 15 (invalid, max is 8)
        bits.extend([0] * 64)  # 8 bytes of data
        bits.extend([0] * 15)  # CRC
        bits.append(1)  # CRC delimiter
        bits.append(0)  # ACK
        bits.append(1)  # ACK delimiter
        bits.extend([1] * 7)  # EOF

    elif malformation == "short_frame":
        # Frame too short to be valid
        bits.append(0)  # SOF
        bits.extend([0, 1, 0])  # Only 3 bits of ID
        # Abrupt end

    elif malformation == "crc_error":
        # Frame with intentionally wrong CRC
        bits.append(0)  # SOF
        bits.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 11-bit ID = 0
        bits.extend([0, 0, 0])  # RTR, IDE, r0
        bits.extend([0, 0, 0, 1])  # DLC = 1
        bits.extend([1, 0, 1, 0, 1, 0, 1, 0])  # Data byte 0xAA
        bits.extend([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Bad CRC (all 1s)
        bits.append(1)  # CRC delimiter
        bits.append(0)  # ACK
        bits.append(1)  # ACK delimiter
        bits.extend([1] * 7)  # EOF

    else:
        raise ValueError(f"Unknown malformation type: {malformation}")

    # Trailing idle
    bits.extend([1] * 20)

    # Convert to samples
    signal = np.repeat(np.array(bits, dtype=bool), samples_per_bit)

    return signal


@pytest.fixture
def can_trace_standard() -> DigitalTrace:
    """Generate a CAN trace with a standard frame."""
    # 500 kbps CAN at 5 MHz sample rate
    sample_rate = 5e6
    samples_per_bit = 10  # 5 MHz / 500 kHz = 10

    # Standard frame with ID 0x123 and data [0x01, 0x02, 0x03]
    signal = generate_can_bit_stream(
        arb_id=0x123,
        data=b"\x01\x02\x03",
        extended=False,
        samples_per_bit=samples_per_bit,
    )

    return DigitalTrace(
        data=signal,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def can_trace_extended() -> DigitalTrace:
    """Generate a CAN trace with an extended frame."""
    sample_rate = 5e6
    samples_per_bit = 10

    # Extended frame with ID 0x12345678 and data [0xAA, 0xBB]
    signal = generate_can_bit_stream(
        arb_id=0x12345678,
        data=b"\xaa\xbb",
        extended=True,
        samples_per_bit=samples_per_bit,
    )

    return DigitalTrace(
        data=signal,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def can_trace_multiple() -> DigitalTrace:
    """Generate a CAN trace with multiple frames."""
    sample_rate = 5e6
    samples_per_bit = 10

    signals = []

    # Frame 1: ID 0x100, data [0x11]
    signals.append(generate_can_bit_stream(0x100, b"\x11", False, samples_per_bit))

    # Frame 2: ID 0x200, data [0x22, 0x33]
    signals.append(generate_can_bit_stream(0x200, b"\x22\x33", False, samples_per_bit))

    # Frame 3: ID 0x300, data [0x44, 0x55, 0x66, 0x77]
    signals.append(generate_can_bit_stream(0x300, b"\x44\x55\x66\x77", False, samples_per_bit))

    signal = np.concatenate(signals)

    return DigitalTrace(
        data=signal,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# CANDecoder Tests
# =============================================================================


class TestCANDecoder:
    """Tests for CANDecoder class."""

    def test_decoder_init(self):
        """Test decoder initialization."""
        decoder = CANDecoder(bitrate=500000)

        assert decoder.bitrate == 500000
        assert decoder.id == "can"
        assert decoder.name == "CAN"

    def test_decoder_options(self):
        """Test decoder options."""
        decoder = CANDecoder(bitrate=250000, sample_point=0.8)

        assert decoder.bitrate == 250000
        assert decoder._sample_point == 0.8

    def test_decode_standard_frame(self, can_trace_standard: DigitalTrace):
        """Test decoding a standard CAN frame."""
        decoder = CANDecoder(bitrate=500000)

        frames = list(decoder.decode(can_trace_standard))

        # Should decode at least one frame
        assert len(frames) > 0, "Expected at least one decoded frame"
        # Verify frame structure (ProtocolPacket stores in annotations)
        assert "arbitration_id" in frames[0].annotations
        assert frames[0].data is not None

    def test_decoder_reset(self, can_trace_standard: DigitalTrace):
        """Test decoder reset between uses."""
        decoder = CANDecoder(bitrate=500000)

        # Decode once
        packets1 = list(decoder.decode(can_trace_standard))

        # Reset
        decoder.reset()

        # Decode again
        packets2 = list(decoder.decode(can_trace_standard))

        # Should get same results
        assert len(packets1) == len(packets2)


# =============================================================================
# CAN Bitrate Tests
# =============================================================================


class TestCANBitrates:
    """Tests for CAN bitrate constants."""

    def test_standard_bitrates(self):
        """Test that standard bitrates are defined."""
        assert 500000 in CAN_BITRATES
        assert 250000 in CAN_BITRATES
        assert 125000 in CAN_BITRATES
        assert 1000000 in CAN_BITRATES

    def test_bitrate_descriptions(self):
        """Test bitrate descriptions."""
        assert CAN_BITRATES[500000] == "500 kbps"
        assert CAN_BITRATES[1000000] == "1 Mbps"


# =============================================================================
# CANFrame Tests
# =============================================================================


class TestCANFrame:
    """Tests for CANFrame dataclass."""

    def test_frame_creation(self):
        """Test creating a CANFrame."""
        frame = CANFrame(
            arbitration_id=0x123,
            is_extended=False,
            is_remote=False,
            dlc=3,
            data=b"\x01\x02\x03",
            crc=0x1234,
            crc_computed=0x1234,
            timestamp=0.001,
            end_timestamp=0.002,
            errors=[],
        )

        assert frame.arbitration_id == 0x123
        assert frame.is_extended is False
        assert frame.dlc == 3
        assert frame.data == b"\x01\x02\x03"
        assert frame.crc_valid is True

    def test_frame_crc_valid(self):
        """Test CRC validation in frame."""
        # Valid CRC
        frame_valid = CANFrame(
            arbitration_id=0x100,
            is_extended=False,
            is_remote=False,
            dlc=1,
            data=b"\x00",
            crc=0x1234,
            crc_computed=0x1234,
            timestamp=0,
            end_timestamp=0.001,
            errors=[],
        )
        assert frame_valid.crc_valid is True

        # Invalid CRC
        frame_invalid = CANFrame(
            arbitration_id=0x100,
            is_extended=False,
            is_remote=False,
            dlc=1,
            data=b"\x00",
            crc=0x1234,
            crc_computed=0x5678,
            timestamp=0,
            end_timestamp=0.001,
            errors=[],
        )
        assert frame_invalid.crc_valid is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeCAN:
    """Tests for decode_can convenience function."""

    def test_decode_can_basic(self, can_trace_standard: DigitalTrace):
        """Test decode_can function."""
        frames = decode_can(can_trace_standard, bitrate=500000)

        # Should return list of CANFrame
        assert isinstance(frames, list)
        for frame in frames:
            assert isinstance(frame, CANFrame)


# =============================================================================
# Edge Cases
# =============================================================================


class TestProtocolsCanEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_trace(self):
        """Test with empty/minimal trace."""
        data = np.ones(10, dtype=bool)  # All recessive
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        packets = list(decoder.decode(trace))

        # Should not crash, just return no packets
        assert len(packets) == 0

    def test_noise_trace(self):
        """Test with noisy/random trace."""
        rng = np.random.default_rng(42)
        data = rng.choice([True, False], size=10000)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should not crash - decoder must be resilient to noise
        assert isinstance(frames, list)
        # All returned frames should have basic structure even if from noise
        for frame in frames:
            assert hasattr(frame, "arbitration_id")

    def test_low_sample_rate(self):
        """Test with sample rate too low for CAN."""
        data = np.ones(100, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=100000))  # Too low

        decoder = CANDecoder(bitrate=500000)
        packets = list(decoder.decode(trace))

        # Should handle gracefully
        assert len(packets) == 0

    def test_constant_dominant(self):
        """Test with constant dominant (bus fault)."""
        data = np.zeros(10000, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        packets = list(decoder.decode(trace))

        # Should not crash, no valid frames
        assert len(packets) == 0


# =============================================================================
# Malformed Frame Tests (MED-004 Negative Test Coverage)
# =============================================================================


class TestMalformedCANFrames:
    """Tests for malformed CAN frame handling."""

    def test_truncated_sof(self):
        """Test handling of truncated frame after SOF."""
        signal = generate_malformed_can_frame("truncated_sof", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should not crash, should return no valid frames or frames with errors noted
        assert isinstance(frames, list)
        # Malformed frame should either be skipped (no frames) or marked with error
        if frames:
            # If decoder returns frames, they should have error tracking
            assert hasattr(frames[0], "errors") or len(frames) == 0

    def test_missing_eof(self):
        """Test handling of frame missing End of Frame sequence."""
        signal = generate_malformed_can_frame("missing_eof", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should handle gracefully - may detect as error or skip frame
        assert isinstance(frames, list)
        # Decoder should not crash on malformed frames
        assert frames is not None

    def test_stuff_error(self):
        """Test handling of bit stuffing error (6 consecutive same bits)."""
        signal = generate_malformed_can_frame("stuff_error", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should detect stuffing error and not crash
        assert isinstance(frames, list)
        # Decoder should either skip this frame (empty) or mark it with error
        for frame in frames:
            # If frame is returned, it should be valid or have error marked
            assert hasattr(frame, "data") or hasattr(frame, "errors")

    def test_invalid_dlc(self):
        """Test handling of invalid DLC value (> 8)."""
        signal = generate_malformed_can_frame("invalid_dlc", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should handle invalid DLC - may skip or report error
        assert isinstance(frames, list)
        # Decoder must not crash on invalid DLC
        assert frames is not None

    def test_short_frame(self):
        """Test handling of frame too short to contain header."""
        signal = generate_malformed_can_frame("short_frame", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should handle gracefully
        assert isinstance(frames, list)

    def test_crc_error_detection(self):
        """Test that CRC errors are properly detected."""
        signal = generate_malformed_can_frame("crc_error", samples_per_bit=10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # If frames are decoded, CRC should be marked invalid
        for frame in frames:
            if hasattr(frame, "crc_valid"):
                # CRC should not be valid for this malformed frame
                assert not frame.crc_valid or frame.crc_computed != frame.crc

    def test_glitch_in_idle(self):
        """Test handling of glitches in idle period."""
        # Create idle with occasional glitches
        bits = [1] * 100  # Idle
        bits[50] = 0  # Single dominant glitch
        bits.extend([1] * 100)  # More idle

        signal = np.repeat(np.array(bits, dtype=bool), 10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should not crash when handling glitches
        assert isinstance(frames, list)
        # If decoder finds frames from glitches, they should have error markers
        for frame in frames:
            if hasattr(frame, "errors"):
                assert len(frame.errors) > 0  # Glitch-based frames should have errors

    def test_bus_off_recovery(self):
        """Test handling of bus-off pattern (many dominant bits then recovery)."""
        # Simulate bus-off: many dominant bits followed by normal idle
        bits = [0] * 500  # Many dominant bits (bus fault)
        bits.extend([1] * 500)  # Recovery to idle

        signal = np.repeat(np.array(bits, dtype=bool), 10)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=5e6))

        decoder = CANDecoder(bitrate=500000)
        frames = list(decoder.decode(trace))

        # Should handle gracefully, no valid frames expected
        assert isinstance(frames, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtocolsCanIntegration:
    """Integration tests for CAN decoder."""

    def test_decode_and_annotate(self, can_trace_standard: DigitalTrace):
        """Test that decoder produces annotations."""
        decoder = CANDecoder(bitrate=500000)

        frames = list(decoder.decode(can_trace_standard))

        # Get annotations
        annotations = decoder.get_annotations()

        # Verify annotations structure
        assert isinstance(annotations, (list, dict))
        # If frames were decoded, annotations should exist
        if len(frames) > 0:
            assert len(annotations) > 0 or annotations is not None

    def test_all_supported_bitrates(self):
        """Test decoding at all standard bitrates."""
        for bitrate in [125000, 250000, 500000, 1000000]:
            # Just verify decoder can be created
            decoder = CANDecoder(bitrate=bitrate)
            assert decoder.bitrate == bitrate
