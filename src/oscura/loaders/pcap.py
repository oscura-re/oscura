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
    import dpkt  # type: ignore[import-not-found]

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
        with open(path, "rb") as f:
            # Detect file format
            magic = f.read(4)
            f.seek(0)

            magic_int = struct.unpack("<I", magic)[0]

            if magic_int == PCAPNG_MAGIC:
                # PCAPNG format
                try:
                    pcap_reader = dpkt.pcapng.Reader(f)
                except AttributeError:
                    raise LoaderError(  # noqa: B904
                        "PCAPNG support requires newer dpkt version",
                        file_path=str(path),
                        fix_hint="Install dpkt >= 1.9: pip install dpkt>=1.9",
                    )
            else:
                # Standard PCAP format
                pcap_reader = dpkt.pcap.Reader(f)

            packets: list[ProtocolPacket] = []
            link_type = getattr(pcap_reader, "datalink", lambda: 1)()

            for timestamp, raw_data in pcap_reader:
                if max_packets is not None and len(packets) >= max_packets:
                    break

                # Parse Ethernet frame
                annotations: dict[str, Any] = {}
                protocol = "RAW"

                try:
                    if link_type == 1:  # Ethernet
                        eth = dpkt.ethernet.Ethernet(raw_data)
                        annotations["src_mac"] = _format_mac(eth.src)
                        annotations["dst_mac"] = _format_mac(eth.dst)

                        # Parse IP layer
                        if isinstance(eth.data, dpkt.ip.IP):
                            ip = eth.data
                            protocol = "IP"
                            annotations["src_ip"] = _format_ip(ip.src)
                            annotations["dst_ip"] = _format_ip(ip.dst)
                            annotations["layer3_protocol"] = "IP"

                            # Parse transport layer
                            if isinstance(ip.data, dpkt.tcp.TCP):
                                tcp = ip.data
                                protocol = "TCP"
                                annotations["src_port"] = tcp.sport
                                annotations["dst_port"] = tcp.dport
                                annotations["layer4_protocol"] = "TCP"
                                annotations["tcp_flags"] = tcp.flags

                            elif isinstance(ip.data, dpkt.udp.UDP):
                                udp = ip.data
                                protocol = "UDP"
                                annotations["src_port"] = udp.sport
                                annotations["dst_port"] = udp.dport
                                annotations["layer4_protocol"] = "UDP"

                            elif isinstance(ip.data, dpkt.icmp.ICMP):
                                protocol = "ICMP"
                                annotations["layer4_protocol"] = "ICMP"

                        elif isinstance(eth.data, dpkt.ip6.IP6):
                            protocol = "IPv6"
                            annotations["layer3_protocol"] = "IPv6"

                        elif isinstance(eth.data, dpkt.arp.ARP):
                            protocol = "ARP"
                            annotations["layer3_protocol"] = "ARP"

                except Exception:
                    # If parsing fails, store raw data
                    pass

                # Apply protocol filter
                if protocol_filter is not None and (
                    annotations.get("layer3_protocol") != protocol_filter
                    and annotations.get("layer4_protocol") != protocol_filter
                    and protocol != protocol_filter
                ):
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
        with open(path, "rb") as f:
            # Read global header (24 bytes)
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

            if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_NS_LE):
                byte_order = "<"
                nanosecond = magic == PCAP_MAGIC_NS_LE
            elif magic in (PCAP_MAGIC_BE, PCAP_MAGIC_NS_BE):
                byte_order = ">"
                nanosecond = magic == PCAP_MAGIC_NS_BE
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

            # Parse rest of header (version_major, version_minor, thiszone, sigfigs, snaplen, network)
            _, _, _, _, snaplen, link_type = struct.unpack(f"{byte_order}HHiIII", header[4:])

            packets: list[ProtocolPacket] = []

            # Read packets
            while True:
                if max_packets is not None and len(packets) >= max_packets:
                    break

                # Read packet header (16 bytes)
                pkt_header = f.read(16)
                if len(pkt_header) < 16:
                    break

                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{byte_order}IIII", pkt_header)

                # Calculate timestamp
                if nanosecond:
                    timestamp = ts_sec + ts_usec / 1e9
                else:
                    timestamp = ts_sec + ts_usec / 1e6

                # Read packet data
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break

                packet = ProtocolPacket(
                    timestamp=timestamp,
                    protocol="RAW",
                    data=bytes(pkt_data),
                    annotations={"original_length": orig_len},
                )
                packets.append(packet)

            return PcapPacketList(
                packets=packets,
                link_type=link_type,
                snaplen=snaplen,
                source_file=str(path),
            )

    except struct.error as e:
        raise FormatError(
            "Corrupted PCAP file",
            file_path=str(path),
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load PCAP file",
            file_path=str(path),
            details=str(e),
            fix_hint="Install dpkt for full PCAP support: pip install dpkt",
        ) from e


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
