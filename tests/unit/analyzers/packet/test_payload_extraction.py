"""Comprehensive tests for payload extraction framework.

Tests cover:
- PayloadInfo dataclass
- PayloadExtractor basic extraction
- Different return types (bytes, memoryview, numpy)
- Batch extraction with filtering
- Streaming iteration
- Metadata preservation
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np

from oscura.analyzers.packet.payload_extraction import (
    PayloadExtractor,
    PayloadInfo,
)

# =============================================================================
# PayloadInfo Tests
# =============================================================================


def test_payload_info_basic() -> None:
    """Test PayloadInfo creation with basic data."""
    info = PayloadInfo(data=b"test", packet_index=0)

    assert info.data == b"test"
    assert info.packet_index == 0
    assert info.timestamp is None
    assert info.src_ip is None
    assert info.dst_ip is None
    assert info.src_port is None
    assert info.dst_port is None
    assert info.protocol is None
    assert info.is_fragment is False
    assert info.fragment_offset == 0


def test_payload_info_with_metadata() -> None:
    """Test PayloadInfo with full metadata."""
    info = PayloadInfo(
        data=b"payload",
        packet_index=5,
        timestamp=1234.567,
        src_ip="192.168.1.1",
        dst_ip="192.168.1.2",
        src_port=12345,
        dst_port=80,
        protocol="TCP",
        is_fragment=True,
        fragment_offset=1024,
    )

    assert info.data == b"payload"
    assert info.packet_index == 5
    assert info.timestamp == 1234.567
    assert info.src_ip == "192.168.1.1"
    assert info.dst_ip == "192.168.1.2"
    assert info.src_port == 12345
    assert info.dst_port == 80
    assert info.protocol == "TCP"
    assert info.is_fragment is True
    assert info.fragment_offset == 1024


# =============================================================================
# PayloadExtractor Initialization Tests
# =============================================================================


def test_extractor_default_init() -> None:
    """Test extractor initialization with defaults."""
    extractor = PayloadExtractor()

    assert extractor.include_headers is False
    assert extractor.zero_copy is True
    assert extractor.return_type == "bytes"


def test_extractor_custom_init() -> None:
    """Test extractor with custom initialization."""
    extractor = PayloadExtractor(include_headers=True, zero_copy=False, return_type="numpy")

    assert extractor.include_headers is True
    assert extractor.zero_copy is False
    assert extractor.return_type == "numpy"


# =============================================================================
# Single Payload Extraction Tests
# =============================================================================


def test_extract_payload_from_dict() -> None:
    """Test payload extraction from dict packet."""
    extractor = PayloadExtractor()
    packet = {"data": b"hello world"}

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, bytes)
    assert payload == b"hello world"


def test_extract_payload_from_bytes() -> None:
    """Test payload extraction from raw bytes."""
    extractor = PayloadExtractor()
    packet = b"raw packet data"

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, bytes)
    assert payload == b"raw packet data"


def test_extract_payload_empty() -> None:
    """Test extraction of empty payload."""
    extractor = PayloadExtractor()
    packet = {"data": b""}

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, bytes)
    assert payload == b""


def test_extract_payload_dict_with_payload_key() -> None:
    """Test extraction when dict uses 'payload' key."""
    extractor = PayloadExtractor()
    packet = {"payload": b"test data"}

    payload = extractor.extract_payload(packet)

    assert payload == b"test data"


def test_extract_payload_from_list() -> None:
    """Test extraction from list of byte values."""
    extractor = PayloadExtractor()
    packet = {"data": [0x48, 0x65, 0x6C, 0x6C, 0x6F]}  # "Hello"

    payload = extractor.extract_payload(packet)

    assert payload == b"Hello"


def test_extract_payload_from_tuple() -> None:
    """Test extraction from tuple of byte values."""
    extractor = PayloadExtractor()
    packet = {"data": (0x74, 0x65, 0x73, 0x74)}  # "test"

    payload = extractor.extract_payload(packet)

    assert payload == b"test"


# =============================================================================
# Return Type Tests
# =============================================================================


def test_extract_payload_bytes_type() -> None:
    """Test extraction with bytes return type."""
    extractor = PayloadExtractor(return_type="bytes")
    packet = b"test"

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, bytes)
    assert payload == b"test"


def test_extract_payload_memoryview_type() -> None:
    """Test extraction with memoryview return type."""
    extractor = PayloadExtractor(return_type="memoryview")
    packet = b"test"

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, memoryview)
    assert bytes(payload) == b"test"


def test_extract_payload_numpy_type() -> None:
    """Test extraction with numpy return type."""
    extractor = PayloadExtractor(return_type="numpy")
    packet = b"test"

    payload = extractor.extract_payload(packet)

    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.uint8
    assert bytes(payload) == b"test"


# =============================================================================
# Batch Extraction Tests
# =============================================================================


def test_extract_all_payloads_basic() -> None:
    """Test batch extraction from multiple packets."""
    extractor = PayloadExtractor()
    packets = [b"packet1", b"packet2", b"packet3"]

    payloads = extractor.extract_all_payloads(packets)

    assert len(payloads) == 3
    assert all(isinstance(p, PayloadInfo) for p in payloads)
    assert payloads[0].data == b"packet1"
    assert payloads[1].data == b"packet2"
    assert payloads[2].data == b"packet3"
    assert payloads[0].packet_index == 0
    assert payloads[1].packet_index == 1
    assert payloads[2].packet_index == 2


def test_extract_all_payloads_with_metadata() -> None:
    """Test batch extraction with metadata preservation."""
    extractor = PayloadExtractor()
    packets = [
        {
            "data": b"udp1",
            "protocol": "UDP",
            "src_ip": "192.168.1.1",
            "dst_ip": "192.168.1.2",
            "src_port": 5000,
            "dst_port": 5001,
            "timestamp": 1.0,
        },
        {
            "data": b"tcp1",
            "protocol": "TCP",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 443,
            "dst_port": 12345,
            "timestamp": 2.0,
        },
    ]

    payloads = extractor.extract_all_payloads(packets)

    assert len(payloads) == 2

    # Check first payload
    assert payloads[0].data == b"udp1"
    assert payloads[0].protocol == "UDP"
    assert payloads[0].src_ip == "192.168.1.1"
    assert payloads[0].dst_ip == "192.168.1.2"
    assert payloads[0].src_port == 5000
    assert payloads[0].dst_port == 5001
    assert payloads[0].timestamp == 1.0

    # Check second payload
    assert payloads[1].data == b"tcp1"
    assert payloads[1].protocol == "TCP"


def test_extract_all_payloads_protocol_filter() -> None:
    """Test filtering by protocol."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"udp1", "protocol": "UDP"},
        {"data": b"tcp1", "protocol": "TCP"},
        {"data": b"udp2", "protocol": "UDP"},
        {"data": b"tcp2", "protocol": "TCP"},
    ]

    udp_payloads = extractor.extract_all_payloads(packets, protocol="UDP")

    assert len(udp_payloads) == 2
    assert all(p.protocol == "UDP" for p in udp_payloads)
    assert udp_payloads[0].data == b"udp1"
    assert udp_payloads[1].data == b"udp2"


def test_extract_all_payloads_port_filter() -> None:
    """Test filtering by source and destination port."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"pkt1", "protocol": "UDP", "src_port": 5000, "dst_port": 5001},
        {"data": b"pkt2", "protocol": "UDP", "src_port": 5000, "dst_port": 5002},
        {"data": b"pkt3", "protocol": "UDP", "src_port": 6000, "dst_port": 5001},
    ]

    # Filter by src_port only
    filtered = extractor.extract_all_payloads(packets, port_filter=(5000, None))
    assert len(filtered) == 2

    # Filter by dst_port only
    filtered = extractor.extract_all_payloads(packets, port_filter=(None, 5001))
    assert len(filtered) == 2

    # Filter by both
    filtered = extractor.extract_all_payloads(packets, port_filter=(5000, 5001))
    assert len(filtered) == 1
    assert filtered[0].data == b"pkt1"


def test_extract_all_payloads_combined_filters() -> None:
    """Test combining protocol and port filters."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"udp5000", "protocol": "UDP", "src_port": 5000, "dst_port": 5001},
        {"data": b"tcp5000", "protocol": "TCP", "src_port": 5000, "dst_port": 5001},
        {"data": b"udp6000", "protocol": "UDP", "src_port": 6000, "dst_port": 5001},
    ]

    filtered = extractor.extract_all_payloads(packets, protocol="UDP", port_filter=(5000, None))

    assert len(filtered) == 1
    assert filtered[0].data == b"udp5000"


def test_extract_all_payloads_case_insensitive_protocol() -> None:
    """Test protocol filtering is case-insensitive."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"pkt1", "protocol": "udp"},
        {"data": b"pkt2", "protocol": "UDP"},
        {"data": b"pkt3", "protocol": "Udp"},
    ]

    filtered = extractor.extract_all_payloads(packets, protocol="UDP")

    assert len(filtered) == 3


def test_extract_all_payloads_fragments() -> None:
    """Test extraction of fragmented packets."""
    extractor = PayloadExtractor()
    packets = [
        {
            "data": b"frag1",
            "protocol": "IP",
            "is_fragment": True,
            "fragment_offset": 0,
        },
        {
            "data": b"frag2",
            "protocol": "IP",
            "is_fragment": True,
            "fragment_offset": 512,
        },
    ]

    payloads = extractor.extract_all_payloads(packets)

    assert len(payloads) == 2
    assert payloads[0].is_fragment is True
    assert payloads[0].fragment_offset == 0
    assert payloads[1].is_fragment is True
    assert payloads[1].fragment_offset == 512


# =============================================================================
# Streaming Iteration Tests
# =============================================================================


def test_iter_payloads_basic() -> None:
    """Test streaming iteration over payloads."""
    extractor = PayloadExtractor()
    packets = [b"pkt1", b"pkt2", b"pkt3"]

    payloads = list(extractor.iter_payloads(packets))

    assert len(payloads) == 3
    assert payloads[0].data == b"pkt1"
    assert payloads[1].data == b"pkt2"
    assert payloads[2].data == b"pkt3"


def test_iter_payloads_with_metadata() -> None:
    """Test iteration with metadata preservation."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"data1", "protocol": "UDP", "src_ip": "1.1.1.1"},
        {"data": b"data2", "protocol": "TCP", "src_ip": "2.2.2.2"},
    ]

    payloads = list(extractor.iter_payloads(packets))

    assert len(payloads) == 2
    assert payloads[0].protocol == "UDP"
    assert payloads[0].src_ip == "1.1.1.1"
    assert payloads[1].protocol == "TCP"
    assert payloads[1].src_ip == "2.2.2.2"


def test_iter_payloads_lazy() -> None:
    """Test that iteration is truly lazy (doesn't process all at once)."""
    extractor = PayloadExtractor()
    packets = [b"pkt1", b"pkt2", b"pkt3"]

    iterator = extractor.iter_payloads(packets)

    # Get first item
    first = next(iterator)
    assert first.data == b"pkt1"

    # Get second item
    second = next(iterator)
    assert second.data == b"pkt2"


def test_iter_payloads_empty() -> None:
    """Test iteration over empty packet list."""
    extractor = PayloadExtractor()
    packets: list[bytes] = []

    payloads = list(extractor.iter_payloads(packets))

    assert len(payloads) == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_extract_payload_missing_keys() -> None:
    """Test extraction from dict without data/payload keys."""
    extractor = PayloadExtractor()
    packet = {"other_key": "value"}

    payload = extractor.extract_payload(packet)

    # Should return empty bytes
    assert payload == b""


def test_extract_all_payloads_mixed_types() -> None:
    """Test batch extraction with mixed packet types."""
    extractor = PayloadExtractor()
    packets = [
        b"raw_bytes",
        {"data": b"dict_packet"},
        b"more_raw",
    ]

    payloads = extractor.extract_all_payloads(packets)

    assert len(payloads) == 3
    assert payloads[0].data == b"raw_bytes"
    assert payloads[1].data == b"dict_packet"
    assert payloads[2].data == b"more_raw"


def test_extract_all_payloads_empty_list() -> None:
    """Test extraction from empty packet list."""
    extractor = PayloadExtractor()
    packets: list[bytes] = []

    payloads = extractor.extract_all_payloads(packets)

    assert len(payloads) == 0


def test_extract_payload_none_data() -> None:
    """Test extraction when data is None."""
    extractor = PayloadExtractor()
    packet = {"data": None}

    payload = extractor.extract_payload(packet)

    # Should handle None gracefully
    assert payload == b""


def test_extract_all_payloads_filter_no_matches() -> None:
    """Test filtering that returns no matches."""
    extractor = PayloadExtractor()
    packets = [
        {"data": b"tcp1", "protocol": "TCP"},
        {"data": b"tcp2", "protocol": "TCP"},
    ]

    # Filter for UDP (none exist)
    filtered = extractor.extract_all_payloads(packets, protocol="UDP")

    assert len(filtered) == 0


def test_extract_payload_memoryview_conversion() -> None:
    """Test that memoryview is converted to bytes in batch extraction."""
    extractor = PayloadExtractor(return_type="memoryview")
    packets = [b"test"]

    payloads = extractor.extract_all_payloads(packets)

    # Should be converted to bytes in PayloadInfo
    assert isinstance(payloads[0].data, bytes)
    assert payloads[0].data == b"test"


def test_extract_payload_numpy_conversion() -> None:
    """Test that numpy array is converted to bytes in batch extraction."""
    extractor = PayloadExtractor(return_type="numpy")
    packets = [b"test"]

    payloads = extractor.extract_all_payloads(packets)

    # Should be converted to bytes in PayloadInfo
    assert isinstance(payloads[0].data, bytes)
    assert payloads[0].data == b"test"


def test_iter_payloads_memoryview_conversion() -> None:
    """Test memoryview conversion in iteration."""
    extractor = PayloadExtractor(return_type="memoryview")
    packets = [b"test"]

    payloads = list(extractor.iter_payloads(packets))

    assert isinstance(payloads[0].data, bytes)


def test_packet_indices_correct() -> None:
    """Test that packet indices are correctly assigned."""
    extractor = PayloadExtractor()
    packets = [b"a", b"b", b"c", b"d", b"e"]

    payloads = extractor.extract_all_payloads(packets)

    for i, payload in enumerate(payloads):
        assert payload.packet_index == i


def test_extract_payload_large_data() -> None:
    """Test extraction of large payload."""
    extractor = PayloadExtractor()
    large_data = b"x" * (10 * 1024 * 1024)  # 10 MB
    packet = large_data

    payload = extractor.extract_payload(packet)

    assert len(payload) == len(large_data)
    assert payload == large_data
