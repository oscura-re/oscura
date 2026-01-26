"""Comprehensive tests for protocol decoder base classes.

Tests cover:
- AnnotationLevel enum
- Annotation dataclass
- ChannelDef and OptionDef
- DecoderState
- ProtocolDecoder base class
- SyncDecoder helpers
- AsyncDecoder helpers
- Edge cases and error handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.protocols.base import (
    Annotation,
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    DecoderState,
    OptionDef,
    ProtocolDecoder,
    SyncDecoder,
)

if TYPE_CHECKING:
    from oscura.core.types import DigitalTrace, ProtocolPacket

# =============================================================================
# Test Implementation of Abstract Classes
# =============================================================================


class DummyDecoder(ProtocolDecoder):
    """Dummy implementation of ProtocolDecoder for testing."""

    id = "test"
    name = "Test"
    longname = "Test Protocol Decoder"
    desc = "Test decoder for unit tests"

    channels = [
        ChannelDef("data", "DATA", "Data line", required=True),
    ]

    optional_channels = [
        ChannelDef("clk", "CLK", "Clock line", required=False),
    ]

    options = [
        OptionDef("baudrate", "Baud rate", default=9600),
        OptionDef("parity", "Parity", default="none", values=["none", "even", "odd"]),
    ]

    annotations = [("bits", "Bit values"), ("bytes", "Byte values")]

    def decode(self, trace: DigitalTrace, **channels: NDArray[np.bool_]) -> list[ProtocolPacket]:
        """Dummy decode implementation."""
        # Return empty list for testing
        return []


class DummySyncDecoder(SyncDecoder):
    """Dummy implementation of SyncDecoder for testing."""

    id = "test_sync"
    name = "Test Sync"

    channels = [
        ChannelDef("clk", "CLK", "Clock"),
        ChannelDef("data", "DATA", "Data"),
    ]

    def decode(self, trace: DigitalTrace, **channels: NDArray[np.bool_]) -> list[ProtocolPacket]:
        """Dummy decode implementation."""
        return []


class DummyAsyncDecoder(AsyncDecoder):
    """Dummy implementation of AsyncDecoder for testing."""

    id = "test_async"
    name = "Test Async"

    channels = [ChannelDef("rx", "RX", "Receive data")]

    def decode(self, trace: DigitalTrace, **channels: NDArray[np.bool_]) -> list[ProtocolPacket]:
        """Dummy decode implementation."""
        return []


# =============================================================================
# AnnotationLevel Tests
# =============================================================================


def test_annotation_level_values() -> None:
    """Test AnnotationLevel enum values."""
    assert AnnotationLevel.BITS == 0
    assert AnnotationLevel.BYTES == 1
    assert AnnotationLevel.WORDS == 2
    assert AnnotationLevel.FIELDS == 3
    assert AnnotationLevel.PACKETS == 4
    assert AnnotationLevel.MESSAGES == 5


def test_annotation_level_ordering() -> None:
    """Test that annotation levels are ordered correctly."""
    assert AnnotationLevel.BITS < AnnotationLevel.BYTES
    assert AnnotationLevel.BYTES < AnnotationLevel.PACKETS
    assert AnnotationLevel.PACKETS < AnnotationLevel.MESSAGES


# =============================================================================
# Annotation Tests
# =============================================================================


def test_annotation_basic() -> None:
    """Test basic annotation creation."""
    ann = Annotation(start_time=1.0, end_time=2.0, level=AnnotationLevel.BITS, text="0b10101010")

    assert ann.start_time == 1.0
    assert ann.end_time == 2.0
    assert ann.level == AnnotationLevel.BITS
    assert ann.text == "0b10101010"
    assert ann.data is None
    assert ann.metadata == {}


def test_annotation_with_data() -> None:
    """Test annotation with binary data."""
    ann = Annotation(
        start_time=1.0,
        end_time=2.0,
        level=AnnotationLevel.BYTES,
        text="0xAA",
        data=b"\xaa",
    )

    assert ann.data == b"\xaa"


def test_annotation_with_metadata() -> None:
    """Test annotation with metadata."""
    ann = Annotation(
        start_time=1.0,
        end_time=2.0,
        level=AnnotationLevel.PACKETS,
        text="UART frame",
        metadata={"baudrate": 9600, "parity": "none"},
    )

    assert ann.metadata["baudrate"] == 9600
    assert ann.metadata["parity"] == "none"


# =============================================================================
# ChannelDef Tests
# =============================================================================


def test_channel_def_basic() -> None:
    """Test basic channel definition."""
    ch = ChannelDef("rx", "RX", "Receive line")

    assert ch.id == "rx"
    assert ch.name == "RX"
    assert ch.desc == "Receive line"
    assert ch.required is True


def test_channel_def_optional() -> None:
    """Test optional channel definition."""
    ch = ChannelDef("cs", "CS#", "Chip select", required=False)

    assert ch.id == "cs"
    assert ch.required is False


# =============================================================================
# OptionDef Tests
# =============================================================================


def test_option_def_basic() -> None:
    """Test basic option definition."""
    opt = OptionDef("baudrate", "Baud rate", default=9600)

    assert opt.id == "baudrate"
    assert opt.name == "Baud rate"
    assert opt.default == 9600
    assert opt.values is None


def test_option_def_with_values() -> None:
    """Test option with enumerated values."""
    opt = OptionDef("parity", "Parity", default="none", values=["none", "even", "odd"])

    assert opt.id == "parity"
    assert opt.values == ["none", "even", "odd"]


# =============================================================================
# DecoderState Tests
# =============================================================================


def test_decoder_state_init() -> None:
    """Test decoder state initialization."""
    state = DecoderState()

    # Should initialize and reset without error
    state.reset()

    # Verify state was initialized correctly
    assert hasattr(state, "reset")
    assert callable(state.reset)


# =============================================================================
# ProtocolDecoder Tests
# =============================================================================


def test_decoder_init_default() -> None:
    """Test decoder initialization with defaults."""
    decoder = DummyDecoder()

    assert decoder.get_option("baudrate") == 9600
    assert decoder.get_option("parity") == "none"


def test_decoder_init_with_options() -> None:
    """Test decoder initialization with custom options."""
    decoder = DummyDecoder(baudrate=115200, parity="even")

    assert decoder.get_option("baudrate") == 115200
    assert decoder.get_option("parity") == "even"


def test_decoder_init_unknown_option() -> None:
    """Test that unknown options raise ValueError."""
    with pytest.raises(ValueError, match="Unknown option: invalid"):
        DummyDecoder(invalid="value")


def test_decoder_set_option() -> None:
    """Test setting option after initialization."""
    decoder = DummyDecoder()

    decoder.set_option("baudrate", 115200)

    assert decoder.get_option("baudrate") == 115200


def test_decoder_reset() -> None:
    """Test decoder reset clears state."""
    decoder = DummyDecoder()

    # Add some annotations
    decoder.put_annotation(0.0, 1.0, AnnotationLevel.BITS, "bit")
    decoder.put_packet(0.0, b"test")

    assert len(decoder.get_annotations()) == 1
    assert len(decoder.get_packets()) == 1

    # Reset
    decoder.reset()

    assert len(decoder.get_annotations()) == 0
    assert len(decoder.get_packets()) == 0


def test_decoder_put_annotation() -> None:
    """Test adding annotations."""
    decoder = DummyDecoder()

    decoder.put_annotation(0.0, 1.0, AnnotationLevel.BITS, "0b01010101", data=b"\x55", flag=True)

    annotations = decoder.get_annotations()
    assert len(annotations) == 1

    ann = annotations[0]
    assert ann.start_time == 0.0
    assert ann.end_time == 1.0
    assert ann.level == AnnotationLevel.BITS
    assert ann.text == "0b01010101"
    assert ann.data == b"\x55"
    assert ann.metadata["flag"] is True


def test_decoder_put_packet() -> None:
    """Test adding decoded packets."""
    decoder = DummyDecoder()

    decoder.put_packet(1.234, b"hello", annotations={"type": "data"}, errors=["checksum"])

    packets = decoder.get_packets()
    assert len(packets) == 1

    pkt = packets[0]
    assert pkt.timestamp == 1.234
    assert pkt.protocol == "test"
    assert pkt.data == b"hello"
    assert pkt.annotations["type"] == "data"
    assert "checksum" in pkt.errors


def test_decoder_get_annotations_filter_level() -> None:
    """Test filtering annotations by level."""
    decoder = DummyDecoder()

    decoder.put_annotation(0.0, 1.0, AnnotationLevel.BITS, "bit1")
    decoder.put_annotation(1.0, 2.0, AnnotationLevel.BYTES, "byte1")
    decoder.put_annotation(2.0, 3.0, AnnotationLevel.BITS, "bit2")

    bits = decoder.get_annotations(level=AnnotationLevel.BITS)
    assert len(bits) == 2
    assert all(a.level == AnnotationLevel.BITS for a in bits)


def test_decoder_get_annotations_filter_time_range() -> None:
    """Test filtering annotations by time range."""
    decoder = DummyDecoder()

    decoder.put_annotation(0.0, 1.0, AnnotationLevel.BITS, "ann1")
    decoder.put_annotation(1.0, 2.0, AnnotationLevel.BITS, "ann2")
    decoder.put_annotation(2.0, 3.0, AnnotationLevel.BITS, "ann3")
    decoder.put_annotation(3.0, 4.0, AnnotationLevel.BITS, "ann4")

    # Get annotations that end after 1.5 (ann2, ann3, ann4)
    filtered = decoder.get_annotations(start_time=1.5)
    assert len(filtered) == 3

    # Get annotations that start before 2.5 (ann1, ann2, ann3)
    filtered = decoder.get_annotations(end_time=2.5)
    assert len(filtered) == 3

    # Get annotations in range [1.0, 3.0] (all 4, as filtering is inclusive)
    filtered = decoder.get_annotations(start_time=1.0, end_time=3.0)
    assert len(filtered) == 4


def test_decoder_get_channel_ids() -> None:
    """Test getting channel IDs."""
    ids = DummyDecoder.get_channel_ids()
    assert ids == ["data"]

    ids_with_optional = DummyDecoder.get_channel_ids(include_optional=True)
    assert "data" in ids_with_optional
    assert "clk" in ids_with_optional


def test_decoder_get_option_ids() -> None:
    """Test getting option IDs."""
    ids = DummyDecoder.get_option_ids()
    assert "baudrate" in ids
    assert "parity" in ids


# =============================================================================
# SyncDecoder Tests
# =============================================================================


def test_sync_decoder_sample_on_rising_edge() -> None:
    """Test sampling data on rising clock edges."""
    decoder = DummySyncDecoder()

    # Clock: 0 1 0 1 0 1
    clock = np.array([False, True, False, True, False, True])
    # Data:  0 1 1 0 0 1
    data = np.array([False, True, True, False, False, True])

    sampled = decoder.sample_on_edge(clock, data, edge="rising")

    # Rising edges at indices 0->1, 2->3, 4->5
    # Sample data at indices 1, 3, 5
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(sampled, expected)


def test_sync_decoder_sample_on_falling_edge() -> None:
    """Test sampling data on falling clock edges."""
    decoder = DummySyncDecoder()

    # Clock: 1 0 1 0 1 0
    clock = np.array([True, False, True, False, True, False])
    # Data:  0 1 1 0 0 1
    data = np.array([False, True, True, False, False, True])

    sampled = decoder.sample_on_edge(clock, data, edge="falling")

    # Falling edges at indices 0->1, 2->3, 4->5
    # Sample data at indices 1, 3, 5
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(sampled, expected)


def test_sync_decoder_no_edges() -> None:
    """Test sampling when no edges present."""
    decoder = DummySyncDecoder()

    # Constant clock (no edges)
    clock = np.array([True, True, True, True])
    data = np.array([False, True, False, True])

    sampled = decoder.sample_on_edge(clock, data, edge="rising")

    assert len(sampled) == 0


# =============================================================================
# AsyncDecoder Tests
# =============================================================================


def test_async_decoder_init() -> None:
    """Test async decoder initialization."""
    decoder = DummyAsyncDecoder(baudrate=115200)

    assert decoder.baudrate == 115200


def test_async_decoder_baudrate_property() -> None:
    """Test baudrate getter and setter."""
    decoder = DummyAsyncDecoder(baudrate=9600)

    assert decoder.baudrate == 9600

    decoder.baudrate = 115200
    assert decoder.baudrate == 115200


def test_async_decoder_bit_time() -> None:
    """Test bit time calculation."""
    decoder = DummyAsyncDecoder(baudrate=9600)

    # At 96 kHz sample rate, 9600 baud = 10 samples per bit
    bit_time = decoder.bit_time(sample_rate=96000)
    assert bit_time == pytest.approx(10.0)


def test_async_decoder_find_start_bit_idle_high() -> None:
    """Test finding start bit with idle high (standard UART)."""
    decoder = DummyAsyncDecoder()

    # Idle high, then falling edge (start bit)
    # Data: 1 1 1 0 0 1 1
    data = np.array([True, True, True, False, False, True, True])

    start_idx = decoder.find_start_bit(data, start_idx=0, idle_high=True)

    # Start bit at index 2->3 (falling edge)
    assert start_idx == 2


def test_async_decoder_find_start_bit_idle_low() -> None:
    """Test finding start bit with idle low (inverted UART)."""
    decoder = DummyAsyncDecoder()

    # Idle low, then rising edge (start bit)
    # Data: 0 0 0 1 1 0 0
    data = np.array([False, False, False, True, True, False, False])

    start_idx = decoder.find_start_bit(data, start_idx=0, idle_high=False)

    # Start bit at index 2->3 (rising edge)
    assert start_idx == 2


def test_async_decoder_find_start_bit_with_offset() -> None:
    """Test finding start bit with offset."""
    decoder = DummyAsyncDecoder()

    data = np.array([True, True, True, False, True, True, False, False])

    # Start search from index 5
    start_idx = decoder.find_start_bit(data, start_idx=5, idle_high=True)

    # Should find edge at 5->6
    assert start_idx == 5


def test_async_decoder_find_start_bit_not_found() -> None:
    """Test finding start bit when none exists."""
    decoder = DummyAsyncDecoder()

    # No transitions
    data = np.array([True, True, True, True])

    start_idx = decoder.find_start_bit(data, start_idx=0, idle_high=True)

    assert start_idx is None


# =============================================================================
# Edge Cases
# =============================================================================


def test_decoder_get_annotations_empty() -> None:
    """Test getting annotations when none exist."""
    decoder = DummyDecoder()

    annotations = decoder.get_annotations()

    assert len(annotations) == 0


def test_decoder_get_packets_empty() -> None:
    """Test getting packets when none exist."""
    decoder = DummyDecoder()

    packets = decoder.get_packets()

    assert len(packets) == 0


def test_decoder_put_packet_minimal() -> None:
    """Test adding packet with minimal arguments."""
    decoder = DummyDecoder()

    decoder.put_packet(1.0, b"data")

    packets = decoder.get_packets()
    assert len(packets) == 1
    assert packets[0].data == b"data"
    assert packets[0].annotations == {}
    assert packets[0].errors == []


def test_sync_decoder_sample_edge_at_boundary() -> None:
    """Test sampling when edge is at array boundary."""
    decoder = DummySyncDecoder()

    # Very short arrays
    clock = np.array([False, True])
    data = np.array([False, True])

    sampled = decoder.sample_on_edge(clock, data, edge="rising")

    assert len(sampled) == 1
    assert sampled[0] == True  # noqa: E712


def test_async_decoder_find_start_bit_empty_array() -> None:
    """Test finding start bit in empty array."""
    decoder = DummyAsyncDecoder()

    data = np.array([], dtype=bool)

    start_idx = decoder.find_start_bit(data)

    assert start_idx is None


def test_decoder_class_attributes() -> None:
    """Test decoder class attributes."""
    assert DummyDecoder.id == "test"
    assert DummyDecoder.name == "Test"
    assert DummyDecoder.api_version == 3
    assert "logic" in DummyDecoder.inputs
    assert "packets" in DummyDecoder.outputs
