"""PCAP/PCAPNG packet capture file loader.

This module provides loading of packet capture files using dpkt
when available, with a basic fallback implementation.


Example:
    >>> from oscura.loaders.pcap import load_pcap
    >>> packets = load_pcap("capture.pcap")
    >>> for packet in packets:
    ...     print(f"Time: {packet.timestamp}, Size: {len(packet.data)} bytes")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

# Try to import dpkt for full PCAP support
try:
    import dpkt

    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False


# PCAP file format constants
PCAP_MAGIC_LE = 0xA1B2C3D4
PCAP_MAGIC_BE = 0xD4C3B2A1
PCAP_MAGIC_NS_LE = 0xA1B23C4D  # Nanosecond resolution
PCAP_MAGIC_NS_BE = 0x4D3CB2A1
PCAPNG_MAGIC = 0x0A0D0D0A


@dataclass
class PcapPacketList:
    """Container for PCAP packets with metadata.

    Allows iteration over packets while preserving capture metadata.

    Attributes:
        packets: List of ProtocolPacket objects.
        link_type: Link layer type (e.g., Ethernet = 1).
        snaplen: Maximum capture length per packet.
        source_file: Path to the source PCAP file.
    """

    packets: list[ProtocolPacket]
    link_type: int = 1  # Ethernet
    snaplen: int = 65535
    source_file: str = ""

    def __iter__(self) -> Iterator[ProtocolPacket]:
        """Iterate over packets."""
        return iter(self.packets)

    def __len__(self) -> int:
        """Return number of packets."""
        return len(self.packets)

    def __getitem__(self, index: int) -> ProtocolPacket:
        """Get packet by index."""
        return self.packets[index]

    def filter(
        self,
        protocol: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ) -> list[ProtocolPacket]:
        """Filter packets by criteria.

        Args:
            protocol: Filter by protocol annotation.
            min_size: Minimum packet size in bytes.
            max_size: Maximum packet size in bytes.

        Returns:
            Filtered list of packets.
        """
        result = self.packets

        if protocol is not None:
            result = [
                p
                for p in result
                if p.annotations.get("layer3_protocol") == protocol
                or p.annotations.get("layer4_protocol") == protocol
            ]

        if min_size is not None:
            result = [p for p in result if len(p.data) >= min_size]

        if max_size is not None:
            result = [p for p in result if len(p.data) <= max_size]

        return result


def load_pcap(
    path: str | PathLike[str],
    *,
    protocol_filter: str | None = None,
    max_packets: int | None = None,
) -> PcapPacketList:
    """Load a PCAP or PCAPNG packet capture file.

    Extracts packets with timestamps and optional protocol annotations.
    Uses dpkt library when available for full protocol dissection.

    Args:
        path: Path to the PCAP/PCAPNG file.
        protocol_filter: Optional protocol filter (e.g., "TCP", "UDP").
        max_packets: Maximum number of packets to load.

    Returns:
        PcapPacketList containing packets and capture metadata.

    Raises:
        LoaderError: If the file cannot be loaded.

    Example:
        >>> packets = load_pcap("network.pcap")
        >>> print(f"Captured {len(packets)} packets")
        >>> for pkt in packets[:5]:
        ...     print(f"  {pkt.timestamp:.6f}s: {len(pkt.data)} bytes")

        >>> # Filter by protocol
        >>> tcp_packets = packets.filter(protocol="TCP")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    if DPKT_AVAILABLE:
        return _load_with_dpkt(
            path,
            protocol_filter=protocol_filter,
            max_packets=max_packets,
        )
    else:
        return _load_basic(
            path,
            protocol_filter=protocol_filter,
            max_packets=max_packets,
        )


def _create_pcap_reader(f: Any, path: Path) -> Any:
    """Create appropriate PCAP reader based on file format.

    Args:
        f: File handle.
        path: Path to PCAP file.

    Returns:
        dpkt PCAP or PCAPNG reader.

    Raises:
        LoaderError: If PCAPNG support is unavailable.
    """
    magic = f.read(4)
    f.seek(0)
    magic_int = struct.unpack("<I", magic)[0]

    if magic_int == PCAPNG_MAGIC:
        try:
            return dpkt.pcapng.Reader(f)
        except AttributeError:
            raise LoaderError(
                "PCAPNG support requires newer dpkt version",
                file_path=str(path),
                fix_hint="Install dpkt >= 1.9: pip install dpkt>=1.9",
            )
    else:
        return dpkt.pcap.Reader(f)


def _parse_transport_layer(ip: Any, annotations: dict[str, Any]) -> str:
    """Parse TCP/UDP/ICMP transport layer from IP packet.

    Args:
        ip: dpkt IP object.
        annotations: Annotations dictionary to populate.

    Returns:
        Protocol name ("TCP", "UDP", "ICMP", or "IP").
    """
    if isinstance(ip.data, dpkt.tcp.TCP):
        tcp = ip.data
        annotations["src_port"] = tcp.sport
        annotations["dst_port"] = tcp.dport
        annotations["layer4_protocol"] = "TCP"
        annotations["tcp_flags"] = tcp.flags
        return "TCP"

    elif isinstance(ip.data, dpkt.udp.UDP):
        udp = ip.data
        annotations["src_port"] = udp.sport
        annotations["dst_port"] = udp.dport
        annotations["layer4_protocol"] = "UDP"
        return "UDP"

    elif isinstance(ip.data, dpkt.icmp.ICMP):
        annotations["layer4_protocol"] = "ICMP"
        return "ICMP"

    return "IP"


def _parse_ethernet_frame(raw_data: bytes, link_type: int) -> tuple[str, dict[str, Any]]:
    """Parse Ethernet frame and extract protocol information.

    Args:
        raw_data: Raw packet bytes.
        link_type: Link layer type.

    Returns:
        Tuple of (protocol_name, annotations_dict).
    """
    annotations: dict[str, Any] = {}
    protocol = "RAW"

    try:
        if link_type != 1:  # Not Ethernet
            return protocol, annotations

        eth = dpkt.ethernet.Ethernet(raw_data)
        annotations["src_mac"] = _format_mac(eth.src)
        annotations["dst_mac"] = _format_mac(eth.dst)

        # Parse network layer
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            annotations["src_ip"] = _format_ip(ip.src)
            annotations["dst_ip"] = _format_ip(ip.dst)
            annotations["layer3_protocol"] = "IP"
            protocol = _parse_transport_layer(ip, annotations)

        elif isinstance(eth.data, dpkt.ip6.IP6):
            protocol = "IPv6"
            annotations["layer3_protocol"] = "IPv6"

        elif isinstance(eth.data, dpkt.arp.ARP):
            protocol = "ARP"
            annotations["layer3_protocol"] = "ARP"

    except Exception:
        # If parsing fails, return defaults
        pass

    return protocol, annotations


def _matches_protocol_filter(
    protocol: str, annotations: dict[str, Any], protocol_filter: str | None
) -> bool:
    """Check if packet matches protocol filter.

    Args:
        protocol: Packet protocol name.
        annotations: Packet annotations.
        protocol_filter: Filter string.

    Returns:
        True if packet matches filter (or no filter set).
    """
    if protocol_filter is None:
        return True

    return (
        annotations.get("layer3_protocol") == protocol_filter
        or annotations.get("layer4_protocol") == protocol_filter
        or protocol == protocol_filter
    )


def _load_with_dpkt(
    path: Path,
    *,
    protocol_filter: str | None = None,
    max_packets: int | None = None,
) -> PcapPacketList:
    """Load PCAP using dpkt library.

    Args:
        path: Path to the PCAP file.
        protocol_filter: Optional protocol filter.
        max_packets: Maximum packets to load.

    Returns:
        PcapPacketList with parsed packets.

    Raises:
        LoaderError: If file cannot be read or dpkt version is incompatible.
    """
    try:
        with open(path, "rb", buffering=65536) as f:
            pcap_reader = _create_pcap_reader(f, path)
            packets: list[ProtocolPacket] = []
            link_type = getattr(pcap_reader, "datalink", lambda: 1)()

            for timestamp, raw_data in pcap_reader:
                if max_packets is not None and len(packets) >= max_packets:
                    break

                protocol, annotations = _parse_ethernet_frame(raw_data, link_type)

                if not _matches_protocol_filter(protocol, annotations, protocol_filter):
                    continue

                packet = ProtocolPacket(
                    timestamp=float(timestamp),
                    protocol=protocol,
                    data=bytes(raw_data),
                    annotations=annotations,
                )
                packets.append(packet)

            return PcapPacketList(
                packets=packets,
                link_type=link_type,
                source_file=str(path),
            )

    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load PCAP file",
            file_path=str(path),
            details=str(e),
            fix_hint="Ensure the file is a valid PCAP/PCAPNG format.",
        ) from e


def _load_basic(
    path: Path,
    *,
    protocol_filter: str | None = None,
    max_packets: int | None = None,
) -> PcapPacketList:
    """Basic PCAP loader without dpkt.

    Args:
        path: Path to the PCAP file.
        protocol_filter: Optional protocol filter (not supported in basic mode).
        max_packets: Maximum packets to load.

    Returns:
        PcapPacketList with raw packet data.

    Raises:
        FormatError: If file is not a valid PCAP.
        LoaderError: If file cannot be read.
    """
    try:
        with open(path, "rb", buffering=65536) as f:
            byte_order, nanosecond, snaplen, link_type = _parse_pcap_header(f, path)
            packets = _read_pcap_packets(f, byte_order, nanosecond, max_packets)

            return PcapPacketList(
                packets=packets,
                link_type=link_type,
                snaplen=snaplen,
                source_file=str(path),
            )

    except struct.error as e:
        raise FormatError("Corrupted PCAP file", file_path=str(path)) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load PCAP file",
            file_path=str(path),
            details=str(e),
            fix_hint="Install dpkt for full PCAP support: pip install dpkt",
        ) from e


def _parse_pcap_header(f: Any, path: Path) -> tuple[str, bool, int, int]:
    """Parse PCAP global header and return format info."""
    header = f.read(24)
    if len(header) < 24:
        raise FormatError(
            "File too small to be a valid PCAP",
            file_path=str(path),
            expected="At least 24 bytes",
            got=f"{len(header)} bytes",
        )

    # Parse magic number
    magic = struct.unpack("<I", header[:4])[0]
    byte_order, nanosecond = _determine_byte_order(magic, path)

    # Parse rest of header
    _, _, _, _, snaplen, link_type = struct.unpack(f"{byte_order}HHiIII", header[4:])
    return byte_order, nanosecond, snaplen, link_type


def _determine_byte_order(magic: int, path: Path) -> tuple[str, bool]:
    """Determine byte order and timestamp precision from magic number."""
    if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_NS_LE):
        return "<", magic == PCAP_MAGIC_NS_LE
    elif magic in (PCAP_MAGIC_BE, PCAP_MAGIC_NS_BE):
        return ">", magic == PCAP_MAGIC_NS_BE
    elif magic == PCAPNG_MAGIC:
        raise LoaderError(
            "PCAPNG format requires dpkt library",
            file_path=str(path),
            fix_hint="Install dpkt: pip install dpkt",
        )
    else:
        raise FormatError(
            "Invalid PCAP magic number",
            file_path=str(path),
            expected="PCAP magic (0xa1b2c3d4)",
            got=f"0x{magic:08x}",
        )


def _read_pcap_packets(
    f: Any, byte_order: str, nanosecond: bool, max_packets: int | None
) -> list[ProtocolPacket]:
    """Read all packets from PCAP file."""
    DEFAULT_MAX_PACKETS = 1_000_000  # Prevent memory exhaustion on unbounded files

    packets: list[ProtocolPacket] = []
    effective_max = max_packets if max_packets is not None else DEFAULT_MAX_PACKETS

    while True:
        if len(packets) >= effective_max:
            break

        packet = _read_one_packet(f, byte_order, nanosecond)
        if packet is None:
            break

        packets.append(packet)

    return packets


def _read_one_packet(f: Any, byte_order: str, nanosecond: bool) -> ProtocolPacket | None:
    """Read one packet from PCAP file."""
    pkt_header = f.read(16)
    if len(pkt_header) < 16:
        return None

    ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{byte_order}IIII", pkt_header)

    # Calculate timestamp
    timestamp = ts_sec + (ts_usec / 1e9 if nanosecond else ts_usec / 1e6)

    # Read packet data
    pkt_data = f.read(incl_len)
    if len(pkt_data) < incl_len:
        return None

    return ProtocolPacket(
        timestamp=timestamp,
        protocol="RAW",
        data=bytes(pkt_data),
        annotations={"original_length": orig_len},
    )


def _format_mac(mac_bytes: bytes) -> str:
    """Format MAC address bytes to string.

    Args:
        mac_bytes: 6-byte MAC address.

    Returns:
        MAC address string (e.g., "00:11:22:33:44:55").
    """
    return ":".join(f"{b:02x}" for b in mac_bytes)


def _format_ip(ip_bytes: bytes) -> str:
    """Format IPv4 address bytes to string.

    Args:
        ip_bytes: 4-byte IPv4 address.

    Returns:
        IPv4 address string (e.g., "192.168.1.1").
    """
    return ".".join(str(b) for b in ip_bytes)


__all__ = ["PcapPacketList", "load_pcap"]
