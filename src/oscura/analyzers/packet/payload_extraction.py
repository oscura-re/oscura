"""Payload extraction framework for network packets.

RE-PAY-001: Payload Extraction Framework

This module provides payload extraction from PCAP packets with metadata
preservation, filtering, and multiple output formats.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class PayloadInfo:
    """Extracted payload with metadata.

    Implements RE-PAY-001: Payload with preserved metadata.

    Attributes:
        data: Payload bytes.
        packet_index: Index of source packet.
        timestamp: Packet timestamp (optional).
        src_ip: Source IP address (optional).
        dst_ip: Destination IP address (optional).
        src_port: Source port (optional).
        dst_port: Destination port (optional).
        protocol: Protocol name (optional).
        is_fragment: Whether packet is a fragment.
        fragment_offset: Fragment offset if fragmented.
    """

    data: bytes
    packet_index: int
    timestamp: float | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None
    protocol: str | None = None
    is_fragment: bool = False
    fragment_offset: int = 0


class PayloadExtractor:
    """Extract payloads from network packets.

    Implements RE-PAY-001: Payload Extraction Framework.

    Provides zero-copy payload extraction from UDP/TCP packets
    with metadata preservation and fragment handling.

    Example:
        >>> extractor = PayloadExtractor()
        >>> payloads = extractor.extract_all_payloads(packets, protocol="UDP")
        >>> for p in payloads:
        ...     print(f"{p.src_ip}:{p.src_port} -> {len(p.data)} bytes")
    """

    def __init__(
        self,
        include_headers: bool = False,
        zero_copy: bool = True,
        return_type: Literal["bytes", "memoryview", "numpy"] = "bytes",
    ) -> None:
        """Initialize payload extractor.

        Args:
            include_headers: Include protocol headers in payload.
            zero_copy: Use zero-copy memoryview where possible.
            return_type: Type for returned payload data.
        """
        self.include_headers = include_headers
        self.zero_copy = zero_copy
        self.return_type = return_type

    def extract_payload(
        self,
        packet: dict[str, Any] | bytes,
        layer: Literal["ethernet", "ip", "transport", "application"] = "application",
    ) -> bytes | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Extract payload from a single packet.

        Implements RE-PAY-001: Single packet payload extraction.

        Args:
            packet: Packet data (dict with 'data' key or raw bytes).
            layer: OSI layer to extract from.

        Returns:
            Payload data in requested format.

        Example:
            >>> payload = extractor.extract_payload(packet)
            >>> print(f"Payload: {len(payload)} bytes")
        """
        # Handle different packet formats
        if isinstance(packet, dict):
            raw_data = packet.get("data", packet.get("payload", b""))
            if isinstance(raw_data, list | tuple):
                raw_data = bytes(raw_data)
        else:
            raw_data = packet

        if not raw_data:
            return self._format_output(b"")

        # For raw bytes, return as-is
        if layer == "application":
            return self._format_output(raw_data)

        # Layer-based extraction would require protocol parsing
        # For now, return full data
        return self._format_output(raw_data)

    def extract_all_payloads(
        self,
        packets: Sequence[dict[str, Any] | bytes],
        protocol: str | None = None,
        port_filter: tuple[int | None, int | None] | None = None,
    ) -> list["PayloadInfo"]:
        """Extract payloads from all packets with metadata.

        Implements RE-PAY-001: Batch payload extraction with metadata.

        Args:
            packets: Sequence of packets.
            protocol: Filter by protocol (e.g., "UDP", "TCP").
            port_filter: (src_port, dst_port) filter tuple.

        Returns:
            List of PayloadInfo with extracted data and metadata.

        Example:
            >>> payloads = extractor.extract_all_payloads(packets, protocol="UDP")
            >>> print(f"Extracted {len(payloads)} payloads")
        """
        results = []

        for i, packet in enumerate(packets):
            if isinstance(packet, dict):
                # Extract metadata from dict
                pkt_protocol = packet.get("protocol", "")
                src_port = packet.get("src_port")
                dst_port = packet.get("dst_port")

                # Apply filters
                if protocol and pkt_protocol.upper() != protocol.upper():
                    continue

                if port_filter:
                    if port_filter[0] is not None and src_port != port_filter[0]:
                        continue
                    if port_filter[1] is not None and dst_port != port_filter[1]:
                        continue

                payload = self.extract_payload(packet)
                if isinstance(payload, memoryview | np.ndarray):
                    payload = bytes(payload)

                info = PayloadInfo(
                    data=payload,
                    packet_index=i,
                    timestamp=packet.get("timestamp"),
                    src_ip=packet.get("src_ip"),
                    dst_ip=packet.get("dst_ip"),
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=pkt_protocol,
                    is_fragment=packet.get("is_fragment", False),
                    fragment_offset=packet.get("fragment_offset", 0),
                )
                results.append(info)
            else:
                # Raw bytes
                payload = bytes(packet)
                info = PayloadInfo(data=payload, packet_index=i)
                results.append(info)

        return results

    def iter_payloads(
        self,
        packets: Sequence[dict[str, Any] | bytes],
    ) -> Iterator["PayloadInfo"]:
        """Iterate over payloads for memory-efficient processing.

        Implements RE-PAY-001: Streaming payload iteration.

        Args:
            packets: Sequence of packets.

        Yields:
            PayloadInfo for each packet.
        """
        for i, packet in enumerate(packets):
            payload = self.extract_payload(packet)
            if isinstance(payload, memoryview | np.ndarray):
                payload = bytes(payload)

            if isinstance(packet, dict):
                info = PayloadInfo(
                    data=payload,
                    packet_index=i,
                    timestamp=packet.get("timestamp"),
                    src_ip=packet.get("src_ip"),
                    dst_ip=packet.get("dst_ip"),
                    src_port=packet.get("src_port"),
                    dst_port=packet.get("dst_port"),
                    protocol=packet.get("protocol"),
                )
            else:
                info = PayloadInfo(data=payload, packet_index=i)

            yield info

    def _format_output(
        self, data: bytes
    ) -> bytes | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Format output according to return_type setting."""
        if self.return_type == "bytes":
            return data
        elif self.return_type == "memoryview":
            return memoryview(data)
        # self.return_type == "numpy"
        return np.frombuffer(data, dtype=np.uint8)


__all__ = [
    "PayloadExtractor",
    "PayloadInfo",
]
