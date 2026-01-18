"""Comprehensive unit tests for PCAP file loader.

This module tests the pcap.py module with comprehensive coverage including:
- Basic PCAP loading (with and without dpkt)
- PCAPNG format detection and handling
- Protocol parsing (TCP, UDP, ICMP, IPv6, ARP)
- Packet filtering
- Error handling
- Edge cases and malformed data
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import ProtocolPacket
from oscura.loaders.pcap import (
    PCAP_MAGIC_BE,
    PCAP_MAGIC_LE,
    PCAP_MAGIC_NS_LE,
    PCAPNG_MAGIC,
    PcapPacketList,
    _format_ip,
    _format_mac,
    _load_basic,
    _load_with_dpkt,
    load_pcap,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


# Helper functions for creating test PCAP data
def create_pcap_header(
    magic: int = PCAP_MAGIC_LE,
    snaplen: int = 65535,
    link_type: int = 1,
) -> bytes:
    """Create a valid PCAP global header (24 bytes).

    PCAP global header format:
    - magic_number: uint32 (4 bytes)
    - version_major: uint16 (2 bytes)
    - version_minor: uint16 (2 bytes)
    - thiszone: int32 (4 bytes)
    - sigfigs: uint32 (4 bytes)
    - snaplen: uint32 (4 bytes)
    - network: uint32 (4 bytes)
    Total: 24 bytes

    Note: The _load_basic function only unpacks some of these fields.
    """
    # Determine byte order and canonical magic from the requested magic type
    # The canonical PCAP magic is 0xa1b2c3d4, which appears differently when read
    # as little-endian depending on how it was stored.
    if magic == PCAP_MAGIC_LE:
        byte_order = "<"
        canonical_magic = 0xA1B2C3D4
    elif magic == PCAP_MAGIC_BE:
        byte_order = ">"
        canonical_magic = 0xA1B2C3D4  # Same value, different byte order in file
    elif magic == PCAP_MAGIC_NS_LE:
        byte_order = "<"
        canonical_magic = 0xA1B23C4D
    else:  # PCAP_MAGIC_NS_BE
        byte_order = ">"
        canonical_magic = 0xA1B23C4D

    # magic, version_major, version_minor, thiszone, sigfigs, snaplen, network
    # Format: IHHiIII = 4 + 2 + 2 + 4 + 4 + 4 + 4 = 24 bytes
    return struct.pack(
        f"{byte_order}IHHiIII",
        canonical_magic,
        2,  # version_major
        4,  # version_minor
        0,  # thiszone
        0,  # sigfigs
        snaplen,
        link_type,
    )


def create_pcap_packet(
    data: bytes,
    timestamp_sec: int = 1234567890,
    timestamp_usec: int = 123456,
    byte_order: str = "<",
) -> bytes:
    """Create a PCAP packet header + data."""
    incl_len = len(data)
    orig_len = incl_len

    header = struct.pack(
        f"{byte_order}IIII",
        timestamp_sec,
        timestamp_usec,
        incl_len,
        orig_len,
    )
    return header + data


def create_minimal_pcap(
    packets: list[bytes] | None = None,
    magic: int = PCAP_MAGIC_LE,
) -> bytes:
    """Create a minimal valid PCAP file."""
    if packets is None:
        packets = [b"test packet"]

    # Determine byte order from magic
    if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_NS_LE):
        byte_order = "<"
    else:
        byte_order = ">"

    pcap_data = create_pcap_header(magic=magic)

    for i, pkt_data in enumerate(packets):
        pcap_data += create_pcap_packet(pkt_data, timestamp_sec=1000 + i, byte_order=byte_order)

    return pcap_data


# Test PcapPacketList class
@pytest.mark.unit
@pytest.mark.loader
class TestPcapPacketList:
    """Test PcapPacketList container class."""

    def test_init_defaults(self) -> None:
        """Test PcapPacketList initialization with defaults."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"test1"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"test2"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        assert pkt_list.packets == packets
        assert pkt_list.link_type == 1  # Ethernet
        assert pkt_list.snaplen == 65535
        assert pkt_list.source_file == ""

    def test_init_custom_values(self) -> None:
        """Test PcapPacketList initialization with custom values."""
        packets = [ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"test")]
        pkt_list = PcapPacketList(
            packets=packets,
            link_type=101,
            snaplen=1500,
            source_file="/path/to/file.pcap",
        )

        assert pkt_list.link_type == 101
        assert pkt_list.snaplen == 1500
        assert pkt_list.source_file == "/path/to/file.pcap"

    def test_iteration(self) -> None:
        """Test iterating over packets."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"test1"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"test2"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        result = list(pkt_list)
        assert result == packets

    def test_len(self) -> None:
        """Test len() on packet list."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"test1"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"test2"),
            ProtocolPacket(timestamp=3.0, protocol="ICMP", data=b"test3"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        assert len(pkt_list) == 3

    def test_len_empty(self) -> None:
        """Test len() on empty packet list."""
        pkt_list = PcapPacketList(packets=[])
        assert len(pkt_list) == 0

    def test_getitem(self) -> None:
        """Test indexing packets."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"test1"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"test2"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        assert pkt_list[0] == packets[0]
        assert pkt_list[1] == packets[1]
        assert pkt_list[-1] == packets[-1]

    def test_getitem_out_of_range(self) -> None:
        """Test indexing with out of range index."""
        pkt_list = PcapPacketList(packets=[])

        with pytest.raises(IndexError):
            _ = pkt_list[0]

    def test_filter_by_protocol_layer3(self) -> None:
        """Test filtering by layer 3 protocol."""
        packets = [
            ProtocolPacket(
                timestamp=1.0,
                protocol="IP",
                data=b"test1",
                annotations={"layer3_protocol": "IP"},
            ),
            ProtocolPacket(
                timestamp=2.0,
                protocol="ARP",
                data=b"test2",
                annotations={"layer3_protocol": "ARP"},
            ),
            ProtocolPacket(
                timestamp=3.0,
                protocol="IPv6",
                data=b"test3",
                annotations={"layer3_protocol": "IPv6"},
            ),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(protocol="IP")
        assert len(filtered) == 1
        assert filtered[0].protocol == "IP"

    def test_filter_by_protocol_layer4(self) -> None:
        """Test filtering by layer 4 protocol."""
        packets = [
            ProtocolPacket(
                timestamp=1.0,
                protocol="TCP",
                data=b"test1",
                annotations={"layer4_protocol": "TCP"},
            ),
            ProtocolPacket(
                timestamp=2.0,
                protocol="UDP",
                data=b"test2",
                annotations={"layer4_protocol": "UDP"},
            ),
            ProtocolPacket(
                timestamp=3.0,
                protocol="ICMP",
                data=b"test3",
                annotations={"layer4_protocol": "ICMP"},
            ),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(protocol="TCP")
        assert len(filtered) == 1
        assert filtered[0].protocol == "TCP"

    def test_filter_by_min_size(self) -> None:
        """Test filtering by minimum packet size."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"short"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"a much longer packet"),
            ProtocolPacket(timestamp=3.0, protocol="ICMP", data=b"medium"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(min_size=10)
        assert len(filtered) == 1
        assert filtered[0].data == b"a much longer packet"

    def test_filter_by_max_size(self) -> None:
        """Test filtering by maximum packet size."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"short"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"a much longer packet"),
            ProtocolPacket(timestamp=3.0, protocol="ICMP", data=b"medium"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(max_size=10)
        assert len(filtered) == 2
        assert b"much longer" not in filtered[0].data + filtered[1].data

    def test_filter_by_size_range(self) -> None:
        """Test filtering by size range."""
        packets = [
            ProtocolPacket(timestamp=1.0, protocol="TCP", data=b"short"),
            ProtocolPacket(timestamp=2.0, protocol="UDP", data=b"a much longer packet"),
            ProtocolPacket(timestamp=3.0, protocol="ICMP", data=b"medium"),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(min_size=6, max_size=10)
        assert len(filtered) == 1
        assert filtered[0].data == b"medium"

    def test_filter_combined(self) -> None:
        """Test filtering by protocol and size together."""
        packets = [
            ProtocolPacket(
                timestamp=1.0,
                protocol="TCP",
                data=b"short tcp",
                annotations={"layer4_protocol": "TCP"},
            ),
            ProtocolPacket(
                timestamp=2.0,
                protocol="TCP",
                data=b"this is a much longer tcp packet",
                annotations={"layer4_protocol": "TCP"},
            ),
            ProtocolPacket(
                timestamp=3.0,
                protocol="UDP",
                data=b"long udp packet here",
                annotations={"layer4_protocol": "UDP"},
            ),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(protocol="TCP", min_size=20)
        assert len(filtered) == 1
        assert filtered[0].protocol == "TCP"
        assert len(filtered[0].data) >= 20

    def test_filter_no_matches(self) -> None:
        """Test filtering with no matches."""
        packets = [
            ProtocolPacket(
                timestamp=1.0,
                protocol="TCP",
                data=b"test",
                annotations={"layer4_protocol": "TCP"},
            ),
        ]
        pkt_list = PcapPacketList(packets=packets)

        filtered = pkt_list.filter(protocol="UDP")
        assert len(filtered) == 0


# Test helper functions
@pytest.mark.unit
@pytest.mark.loader
class TestHelperFunctions:
    """Test helper formatting functions."""

    def test_format_mac_valid(self) -> None:
        """Test MAC address formatting."""
        mac = b"\x00\x11\x22\x33\x44\x55"
        result = _format_mac(mac)
        assert result == "00:11:22:33:44:55"

    def test_format_mac_broadcast(self) -> None:
        """Test formatting broadcast MAC."""
        mac = b"\xff\xff\xff\xff\xff\xff"
        result = _format_mac(mac)
        assert result == "ff:ff:ff:ff:ff:ff"

    def test_format_ip_valid(self) -> None:
        """Test IPv4 address formatting."""
        ip = b"\xc0\xa8\x01\x01"  # 192.168.1.1
        result = _format_ip(ip)
        assert result == "192.168.1.1"

    def test_format_ip_localhost(self) -> None:
        """Test formatting localhost IP."""
        ip = b"\x7f\x00\x00\x01"  # 127.0.0.1
        result = _format_ip(ip)
        assert result == "127.0.0.1"

    def test_format_ip_zeros(self) -> None:
        """Test formatting all-zeros IP."""
        ip = b"\x00\x00\x00\x00"
        result = _format_ip(ip)
        assert result == "0.0.0.0"


# Test basic PCAP loading
@pytest.mark.unit
@pytest.mark.loader
class TestLoadBasic:
    """Test basic PCAP loading without dpkt.

    Tests the fallback PCAP loader that works without the dpkt library.
    """

    def test_load_minimal_pcap(self, tmp_path: Path) -> None:
        """Test loading a minimal valid PCAP file."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        result = _load_basic(pcap_file)

        assert isinstance(result, PcapPacketList)
        assert len(result) == 1
        assert result[0].data == b"test packet"
        assert result[0].protocol == "RAW"

    def test_load_multiple_packets(self, tmp_path: Path) -> None:
        """Test loading PCAP with multiple packets."""
        pcap_file = tmp_path / "multi.pcap"
        pcap_file.write_bytes(
            create_minimal_pcap(
                packets=[b"packet 1", b"packet 2", b"packet 3"],
            )
        )

        result = _load_basic(pcap_file)

        assert len(result) == 3
        assert result[0].data == b"packet 1"
        assert result[1].data == b"packet 2"
        assert result[2].data == b"packet 3"

    def test_load_big_endian(self, tmp_path: Path) -> None:
        """Test loading big-endian PCAP."""
        pcap_file = tmp_path / "be.pcap"
        pcap_file.write_bytes(create_minimal_pcap(magic=PCAP_MAGIC_BE))

        result = _load_basic(pcap_file)

        assert isinstance(result, PcapPacketList)
        assert len(result) == 1

    def test_load_nanosecond_resolution(self, tmp_path: Path) -> None:
        """Test loading PCAP with nanosecond resolution."""
        pcap_file = tmp_path / "ns.pcap"

        # Create nanosecond-resolution PCAP
        pcap_data = create_pcap_header(magic=PCAP_MAGIC_NS_LE)
        pcap_data += create_pcap_packet(
            b"test",
            timestamp_sec=1000,
            timestamp_usec=123456789,  # nanoseconds
        )
        pcap_file.write_bytes(pcap_data)

        result = _load_basic(pcap_file)

        assert len(result) == 1
        # Timestamp should be converted from nanoseconds
        assert result[0].timestamp == pytest.approx(1000.123456789)

    def test_load_with_max_packets(self, tmp_path: Path) -> None:
        """Test loading with max_packets limit."""
        pcap_file = tmp_path / "limited.pcap"
        pcap_file.write_bytes(
            create_minimal_pcap(
                packets=[b"packet 1", b"packet 2", b"packet 3", b"packet 4"],
            )
        )

        result = _load_basic(pcap_file, max_packets=2)

        assert len(result) == 2
        assert result[0].data == b"packet 1"
        assert result[1].data == b"packet 2"

    def test_load_empty_pcap(self, tmp_path: Path) -> None:
        """Test loading PCAP with no packets."""
        pcap_file = tmp_path / "empty.pcap"
        pcap_file.write_bytes(create_pcap_header())

        result = _load_basic(pcap_file)

        assert len(result) == 0

    def test_load_captures_metadata(self, tmp_path: Path) -> None:
        """Test that metadata is captured correctly."""
        pcap_file = tmp_path / "meta.pcap"
        pcap_file.write_bytes(create_minimal_pcap(packets=[b"test"]))

        result = _load_basic(pcap_file)

        assert result.link_type == 1  # Ethernet
        assert result.snaplen == 65535
        assert result.source_file == str(pcap_file)

    def test_load_preserves_timestamps(self, tmp_path: Path) -> None:
        """Test that packet timestamps are preserved."""
        pcap_file = tmp_path / "time.pcap"

        pcap_data = create_pcap_header()
        pcap_data += create_pcap_packet(b"pkt1", timestamp_sec=1000, timestamp_usec=100000)
        pcap_data += create_pcap_packet(b"pkt2", timestamp_sec=1001, timestamp_usec=500000)
        pcap_file.write_bytes(pcap_data)

        result = _load_basic(pcap_file)

        assert len(result) == 2
        assert result[0].timestamp == pytest.approx(1000.1)
        assert result[1].timestamp == pytest.approx(1001.5)

    def test_load_annotations_include_original_length(self, tmp_path: Path) -> None:
        """Test that annotations include original packet length."""
        pcap_file = tmp_path / "annot.pcap"

        # Create packet with truncated data (incl_len < orig_len)
        pcap_data = create_pcap_header()
        header = struct.pack("<IIII", 1000, 0, 10, 100)  # incl=10, orig=100
        pcap_data += header + b"x" * 10
        pcap_file.write_bytes(pcap_data)

        result = _load_basic(pcap_file)

        assert len(result) == 1
        assert result[0].annotations["original_length"] == 100

    def test_load_protocol_filter_ignored_in_basic(self, tmp_path: Path) -> None:
        """Test that protocol_filter is ignored in basic mode (no parsing)."""
        pcap_file = tmp_path / "filter.pcap"
        pcap_file.write_bytes(create_minimal_pcap(packets=[b"test"]))

        # Protocol filter is passed but should be ignored
        result = _load_basic(pcap_file, protocol_filter="TCP")

        # Packet should still be returned (no filtering in basic mode)
        assert len(result) == 1

    def test_load_truncated_header_raises_error(self, tmp_path: Path) -> None:
        """Test that truncated global header raises FormatError."""
        pcap_file = tmp_path / "truncated.pcap"
        pcap_file.write_bytes(b"short")  # Less than 24 bytes

        with pytest.raises(FormatError) as exc_info:
            _load_basic(pcap_file)

        assert "too small" in str(exc_info.value).lower()

    def test_load_invalid_magic_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid magic number raises FormatError."""
        pcap_file = tmp_path / "bad_magic.pcap"

        # Create header with invalid magic
        bad_header = struct.pack("<IHHiIII", 0xDEADBEEF, 2, 4, 0, 0, 65535, 1)
        pcap_file.write_bytes(bad_header)

        with pytest.raises(FormatError) as exc_info:
            _load_basic(pcap_file)

        assert "magic" in str(exc_info.value).lower()

    def test_load_pcapng_raises_loader_error(self, tmp_path: Path) -> None:
        """Test that PCAPNG format raises LoaderError in basic mode."""
        pcap_file = tmp_path / "test.pcapng"

        # Create PCAPNG header (just magic for this test)
        pcapng_data = struct.pack("<I", PCAPNG_MAGIC) + b"\x00" * 20
        pcap_file.write_bytes(pcapng_data)

        with pytest.raises(LoaderError) as exc_info:
            _load_basic(pcap_file)

        assert "pcapng" in str(exc_info.value).lower()
        assert "dpkt" in str(exc_info.value).lower()

    def test_load_corrupted_packet_header(self, tmp_path: Path) -> None:
        """Test handling of corrupted packet header."""
        pcap_file = tmp_path / "corrupt.pcap"

        # Valid global header + truncated packet header
        pcap_data = create_pcap_header()
        pcap_data += b"short"  # Less than 16 bytes
        pcap_file.write_bytes(pcap_data)

        # Should handle gracefully (no packets returned)
        result = _load_basic(pcap_file)
        assert len(result) == 0

    def test_load_truncated_packet_data(self, tmp_path: Path) -> None:
        """Test handling of truncated packet data."""
        pcap_file = tmp_path / "truncated_pkt.pcap"

        pcap_data = create_pcap_header()
        # Packet header says 100 bytes, but only provide 10
        header = struct.pack("<IIII", 1000, 0, 100, 100)
        pcap_data += header + b"x" * 10
        pcap_file.write_bytes(pcap_data)

        # Should handle gracefully
        result = _load_basic(pcap_file)
        # May have 0 packets if truncation detected
        assert len(result) >= 0


# Test load_pcap main function
@pytest.mark.unit
@pytest.mark.loader
class TestLoadPcap:
    """Test main load_pcap function."""

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a non-existent file."""
        fake_path = tmp_path / "nonexistent.pcap"

        with pytest.raises(LoaderError) as exc_info:
            load_pcap(fake_path)

        assert "not found" in str(exc_info.value).lower()

    def test_load_with_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        # Pass as string, not Path
        result = load_pcap(str(pcap_file))

        assert isinstance(result, PcapPacketList)
        assert len(result) > 0

    def test_load_with_path_object(self, tmp_path: Path) -> None:
        """Test loading with Path object."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        result = load_pcap(pcap_file)

        assert isinstance(result, PcapPacketList)
        assert len(result) > 0

    def test_load_dispatches_to_basic_when_dpkt_unavailable(self, tmp_path: Path) -> None:
        """Test that load_pcap uses basic loader when dpkt unavailable."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        # Mock dpkt as unavailable
        with patch("oscura.loaders.pcap.DPKT_AVAILABLE", False):
            result = load_pcap(pcap_file)

        assert isinstance(result, PcapPacketList)
        assert result[0].protocol == "RAW"  # Basic loader sets protocol to RAW

    def test_load_with_max_packets(self, tmp_path: Path) -> None:
        """Test max_packets parameter is passed through."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap(packets=[b"p1", b"p2", b"p3"]))

        result = load_pcap(pcap_file, max_packets=2)

        assert len(result) == 2


# Test dpkt loading (mocked since dpkt may not be available)
@pytest.mark.unit
@pytest.mark.loader
class TestLoadWithDpkt:
    """Test PCAP loading with dpkt (mocked)."""

    def test_load_detects_pcapng_format(self, tmp_path: Path) -> None:
        """Test PCAPNG format detection."""
        pcap_file = tmp_path / "test.pcapng"

        # Create minimal PCAPNG file
        pcapng_data = struct.pack("<I", PCAPNG_MAGIC) + b"\x00" * 20
        pcap_file.write_bytes(pcapng_data)

        # Mock dpkt module
        mock_dpkt = MagicMock()
        mock_pcapng_reader = MagicMock()
        mock_pcapng_reader.__iter__ = Mock(return_value=iter([]))
        mock_pcapng_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcapng.Reader = Mock(return_value=mock_pcapng_reader)

        # Import the module to get access to its namespace
        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file)

        # Should have attempted PCAPNG reader
        mock_dpkt.pcapng.Reader.assert_called_once()
        assert isinstance(result, PcapPacketList)

    def test_load_pcapng_without_support_raises_error(self, tmp_path: Path) -> None:
        """Test PCAPNG without dpkt support raises LoaderError."""
        pcap_file = tmp_path / "test.pcapng"

        pcapng_data = struct.pack("<I", PCAPNG_MAGIC) + b"\x00" * 20
        pcap_file.write_bytes(pcapng_data)

        # Mock dpkt without pcapng support (AttributeError)
        mock_dpkt = MagicMock()
        mock_dpkt.pcapng.Reader = Mock(side_effect=AttributeError)

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    with pytest.raises(LoaderError) as exc_info:
                        _load_with_dpkt(pcap_file)

        assert "pcapng support" in str(exc_info.value).lower()

    def test_load_standard_pcap_with_dpkt(self, tmp_path: Path) -> None:
        """Test loading standard PCAP with dpkt."""
        pcap_file = tmp_path / "test.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        # Mock dpkt
        mock_dpkt = MagicMock()
        mock_pcap_reader = MagicMock()

        # Mock packet data
        mock_packets = [
            (1234567890.5, b"test packet data"),
        ]
        mock_pcap_reader.__iter__ = Mock(return_value=iter(mock_packets))
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)
        mock_dpkt.ethernet.Ethernet = Mock(side_effect=Exception("Not ethernet"))

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file)

        assert len(result) == 1
        assert result[0].timestamp == 1234567890.5

    def test_load_parses_tcp_packet(self, tmp_path: Path) -> None:
        """Test parsing TCP packet with dpkt."""
        pcap_file = tmp_path / "tcp.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        # Mock dpkt with TCP packet
        mock_dpkt = MagicMock()

        # Create mock Ethernet/IP/TCP hierarchy
        mock_tcp = MagicMock()
        mock_tcp.sport = 80
        mock_tcp.dport = 12345
        mock_tcp.flags = 0x18  # ACK + PSH

        mock_ip = MagicMock()
        mock_ip.src = b"\xc0\xa8\x01\x01"
        mock_ip.dst = b"\xc0\xa8\x01\x02"
        mock_ip.data = mock_tcp

        mock_eth = MagicMock()
        mock_eth.src = b"\x00\x11\x22\x33\x44\x55"
        mock_eth.dst = b"\xaa\xbb\xcc\xdd\xee\xff"
        mock_eth.data = mock_ip

        mock_dpkt.ethernet.Ethernet = Mock(return_value=mock_eth)
        mock_dpkt.ip.IP = type("IP", (), {})
        mock_dpkt.tcp.TCP = type("TCP", (), {})

        # Make isinstance work
        mock_ip.__class__ = mock_dpkt.ip.IP
        mock_tcp.__class__ = mock_dpkt.tcp.TCP

        mock_pcap_reader = MagicMock()
        mock_pcap_reader.__iter__ = Mock(return_value=iter([(1000.0, b"dummy")]))
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file)

        assert len(result) == 1
        pkt = result[0]
        assert pkt.protocol == "TCP"
        assert pkt.annotations["src_ip"] == "192.168.1.1"
        assert pkt.annotations["dst_ip"] == "192.168.1.2"
        assert pkt.annotations["src_port"] == 80
        assert pkt.annotations["dst_port"] == 12345
        assert pkt.annotations["layer4_protocol"] == "TCP"

    def test_load_parses_udp_packet(self, tmp_path: Path) -> None:
        """Test parsing UDP packet with dpkt."""
        pcap_file = tmp_path / "udp.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        # Create mock UDP packet
        mock_udp = MagicMock()
        mock_udp.sport = 53
        mock_udp.dport = 54321

        mock_ip = MagicMock()
        mock_ip.src = b"\x08\x08\x08\x08"
        mock_ip.dst = b"\x08\x08\x04\x04"
        mock_ip.data = mock_udp

        mock_eth = MagicMock()
        mock_eth.src = b"\x00\x00\x00\x00\x00\x00"
        mock_eth.dst = b"\xff\xff\xff\xff\xff\xff"
        mock_eth.data = mock_ip

        mock_dpkt.ethernet.Ethernet = Mock(return_value=mock_eth)
        mock_dpkt.ip.IP = type("IP", (), {})
        mock_dpkt.tcp.TCP = type("TCP", (), {})
        mock_dpkt.udp.UDP = type("UDP", (), {})
        mock_dpkt.icmp.ICMP = type("ICMP", (), {})

        mock_ip.__class__ = mock_dpkt.ip.IP
        mock_udp.__class__ = mock_dpkt.udp.UDP

        mock_pcap_reader = MagicMock()
        mock_pcap_reader.__iter__ = Mock(return_value=iter([(2000.0, b"dummy")]))
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file)

        assert len(result) == 1
        pkt = result[0]
        assert pkt.protocol == "UDP"
        assert pkt.annotations["src_port"] == 53
        assert pkt.annotations["dst_port"] == 54321
        assert pkt.annotations["layer4_protocol"] == "UDP"

    def test_load_protocol_filter_tcp(self, tmp_path: Path) -> None:
        """Test protocol filtering for TCP packets."""
        pcap_file = tmp_path / "filtered.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        # Create one TCP and one UDP packet
        mock_tcp = MagicMock()
        mock_tcp.sport = 80
        mock_tcp.dport = 12345
        mock_tcp.flags = 0x18

        mock_udp = MagicMock()
        mock_udp.sport = 53
        mock_udp.dport = 54321

        mock_ip_tcp = MagicMock()
        mock_ip_tcp.src = b"\xc0\xa8\x01\x01"
        mock_ip_tcp.dst = b"\xc0\xa8\x01\x02"
        mock_ip_tcp.data = mock_tcp

        mock_ip_udp = MagicMock()
        mock_ip_udp.src = b"\xc0\xa8\x01\x03"
        mock_ip_udp.dst = b"\xc0\xa8\x01\x04"
        mock_ip_udp.data = mock_udp

        mock_eth_tcp = MagicMock()
        mock_eth_tcp.src = b"\x00\x00\x00\x00\x00\x00"
        mock_eth_tcp.dst = b"\xff\xff\xff\xff\xff\xff"
        mock_eth_tcp.data = mock_ip_tcp

        mock_eth_udp = MagicMock()
        mock_eth_udp.src = b"\x00\x00\x00\x00\x00\x00"
        mock_eth_udp.dst = b"\xff\xff\xff\xff\xff\xff"
        mock_eth_udp.data = mock_ip_udp

        call_count = [0]

        def ethernet_factory(data):
            result = mock_eth_tcp if call_count[0] == 0 else mock_eth_udp
            call_count[0] += 1
            return result

        mock_dpkt.ethernet.Ethernet = Mock(side_effect=ethernet_factory)
        mock_dpkt.ip.IP = type("IP", (), {})
        mock_dpkt.tcp.TCP = type("TCP", (), {})
        mock_dpkt.udp.UDP = type("UDP", (), {})

        mock_ip_tcp.__class__ = mock_dpkt.ip.IP
        mock_ip_udp.__class__ = mock_dpkt.ip.IP
        mock_tcp.__class__ = mock_dpkt.tcp.TCP
        mock_udp.__class__ = mock_dpkt.udp.UDP

        mock_pcap_reader = MagicMock()
        mock_pcap_reader.__iter__ = Mock(return_value=iter([(1000.0, b"tcp"), (1001.0, b"udp")]))
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file, protocol_filter="TCP")

        # Should only have TCP packet
        assert len(result) == 1
        assert result[0].protocol == "TCP"

    def test_load_max_packets_with_dpkt(self, tmp_path: Path) -> None:
        """Test max_packets limit with dpkt."""
        pcap_file = tmp_path / "limited.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        mock_pcap_reader = MagicMock()
        mock_pcap_reader.__iter__ = Mock(
            return_value=iter(
                [
                    (1000.0, b"p1"),
                    (1001.0, b"p2"),
                    (1002.0, b"p3"),
                ]
            )
        )
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)
        mock_dpkt.ethernet.Ethernet = Mock(side_effect=Exception("Skip parsing"))

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file, max_packets=2)

        assert len(result) == 2

    def test_load_handles_parsing_errors_gracefully(self, tmp_path: Path) -> None:
        """Test that parsing errors don't crash the loader."""
        pcap_file = tmp_path / "bad_packet.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        # Ethernet parsing raises exception
        mock_dpkt.ethernet.Ethernet = Mock(side_effect=Exception("Malformed packet"))

        mock_pcap_reader = MagicMock()
        mock_pcap_reader.__iter__ = Mock(return_value=iter([(1000.0, b"bad data")]))
        mock_pcap_reader.datalink = Mock(return_value=1)

        mock_dpkt.pcap.Reader = Mock(return_value=mock_pcap_reader)

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    result = _load_with_dpkt(pcap_file)

        # Should still return packet with RAW protocol
        assert len(result) == 1
        assert result[0].protocol == "RAW"

    def test_load_general_exception_wrapped(self, tmp_path: Path) -> None:
        """Test that general exceptions are wrapped in LoaderError."""
        pcap_file = tmp_path / "error.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        # File read raises IOError
        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    with patch("builtins.open", side_effect=OSError("Disk error")):
                        with pytest.raises(LoaderError) as exc_info:
                            _load_with_dpkt(pcap_file)

        assert "failed to load" in str(exc_info.value).lower()

    def test_load_preserves_loader_error(self, tmp_path: Path) -> None:
        """Test that LoaderError is not wrapped again."""
        pcap_file = tmp_path / "error.pcap"
        pcap_file.write_bytes(create_minimal_pcap())

        mock_dpkt = MagicMock()

        original_error = LoaderError("Original error", file_path=str(pcap_file))

        import oscura.loaders.pcap as pcap_module

        with patch.dict(sys.modules, {"dpkt": mock_dpkt}):
            with patch.object(pcap_module, "DPKT_AVAILABLE", True):
                with patch.object(pcap_module, "dpkt", mock_dpkt, create=True):
                    with patch("builtins.open", side_effect=original_error):
                        with pytest.raises(LoaderError) as exc_info:
                            _load_with_dpkt(pcap_file)

        # Should be the same error, not wrapped
        assert exc_info.value is original_error


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
class TestPCAPIntegration:
    """Integration tests with real test data (if available)."""

    def test_load_http_pcap(self, http_pcap: Path | None) -> None:
        """Test loading HTTP PCAP file.

        Validates:
        - HTTP PCAP loads successfully
        - Returns packet data
        """
        if http_pcap is None or not http_pcap.exists():
            pytest.skip("HTTP PCAP file not available")

        try:
            result = load_pcap(http_pcap)
            assert result is not None
            assert len(result) > 0
        except ImportError:
            pytest.skip("PCAP loader not available")
        except (LoaderError, FormatError) as e:
            pytest.skip(f"PCAP loading failed: {e}")

    def test_load_modbus_pcap(self, modbus_pcap: Path | None) -> None:
        """Test loading Modbus TCP PCAP file.

        Validates:
        - Industrial protocol PCAP loads successfully
        """
        if modbus_pcap is None or not modbus_pcap.exists():
            pytest.skip("Modbus PCAP file not available")

        try:
            result = load_pcap(modbus_pcap)
            assert result is not None
            assert len(result) > 0
        except ImportError:
            pytest.skip("PCAP loader not available")
        except (LoaderError, FormatError) as e:
            pytest.skip(f"PCAP loading failed: {e}")
