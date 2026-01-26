"""Zigbee protocol analyzer with network topology discovery.

This module provides comprehensive Zigbee protocol analysis including
NWK layer parsing, APS layer decoding, ZCL cluster support, and
network topology discovery.

Example:
    >>> from oscura.iot.zigbee import ZigbeeAnalyzer, ZigbeeFrame
    >>> analyzer = ZigbeeAnalyzer()
    >>> frame = ZigbeeFrame(
    ...     timestamp=0.0,
    ...     frame_type="DATA",
    ...     source_address=0x1234,
    ...     dest_address=0x0000,
    ... )
    >>> analyzer.add_frame(frame)
    >>> topology = analyzer.discover_topology()

References:
    Zigbee Specification (CSA-IOT)
    Zigbee NWK Layer Specification
    Zigbee APS Layer Specification
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.iot.zigbee.security import parse_security_header
from oscura.iot.zigbee.zcl import ZCL_CLUSTERS, parse_zcl_frame


@dataclass
class ZigbeeFrame:
    """Zigbee frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        frame_type: Frame type ("DATA", "COMMAND", "ACK").
        source_address: 16-bit network address of source.
        dest_address: 16-bit network address of destination.
        source_ieee: 64-bit IEEE address of source (optional).
        dest_ieee: 64-bit IEEE address of destination (optional).
        sequence_number: NWK sequence number.
        radius: Maximum hop count.
        payload: Frame payload bytes.
        decoded_aps: Decoded APS layer data (optional).
        decoded_zcl: Decoded ZCL data (optional).

    Example:
        >>> frame = ZigbeeFrame(
        ...     timestamp=1.0,
        ...     frame_type="DATA",
        ...     source_address=0x1234,
        ...     dest_address=0x0000,
        ...     payload=b"\\x00\\x01\\x02",
        ... )
    """

    timestamp: float
    frame_type: str
    source_address: int
    dest_address: int
    source_ieee: int | None = None
    dest_ieee: int | None = None
    sequence_number: int = 0
    radius: int = 0
    payload: bytes = b""
    decoded_aps: dict[str, Any] | None = None
    decoded_zcl: dict[str, Any] | None = None


@dataclass
class ZigbeeDevice:
    """Zigbee device in network.

    Attributes:
        short_address: 16-bit network address.
        ieee_address: 64-bit IEEE MAC address (optional).
        device_type: Device type ("coordinator", "router", "end_device").
        parent_address: Parent device address for routing (optional).
        clusters: List of supported cluster IDs.
        manufacturer: Manufacturer name (optional).
        model: Model identifier (optional).

    Example:
        >>> device = ZigbeeDevice(
        ...     short_address=0x1234,
        ...     ieee_address=0x0013A20040A12345,
        ...     device_type="end_device",
        ...     clusters=[0x0006, 0x0008],
        ... )
    """

    short_address: int
    ieee_address: int | None = None
    device_type: str = "unknown"
    parent_address: int | None = None
    clusters: list[int] = field(default_factory=list)
    manufacturer: str | None = None
    model: str | None = None


class ZigbeeAnalyzer:
    """Zigbee protocol analyzer with ZCL cluster support.

    Analyzes Zigbee network traffic including NWK/APS/ZCL layers,
    discovers network topology, and supports security frame parsing.

    Attributes:
        STANDARD_CLUSTERS: Standard ZCL cluster ID to name mapping.
        FRAME_TYPES: NWK frame type codes.

    Example:
        >>> analyzer = ZigbeeAnalyzer()
        >>> analyzer.add_frame(frame)
        >>> topology = analyzer.discover_topology()
        >>> analyzer.export_topology(Path("network.json"))
    """

    # Standard ZCL cluster IDs
    STANDARD_CLUSTERS: ClassVar[dict[int, str]] = ZCL_CLUSTERS

    # Frame types
    FRAME_TYPES: ClassVar[dict[int, str]] = {
        0x00: "DATA",
        0x01: "COMMAND",
        0x02: "ACK",
    }

    def __init__(self) -> None:
        """Initialize Zigbee analyzer.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> len(analyzer.frames)
            0
        """
        self.frames: list[ZigbeeFrame] = []
        self.devices: dict[int, ZigbeeDevice] = {}
        self.network_keys: list[bytes] = []

    def add_frame(self, frame: ZigbeeFrame) -> None:
        """Add Zigbee frame for analysis.

        Args:
            frame: Zigbee frame to analyze.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> frame = ZigbeeFrame(
            ...     timestamp=0.0,
            ...     frame_type="DATA",
            ...     source_address=0x1234,
            ...     dest_address=0x0000,
            ... )
            >>> analyzer.add_frame(frame)
            >>> len(analyzer.frames)
            1
        """
        self.frames.append(frame)

        # Update device registry
        if frame.source_address not in self.devices:
            self.devices[frame.source_address] = ZigbeeDevice(
                short_address=frame.source_address,
                ieee_address=frame.source_ieee,
            )
        elif frame.source_ieee and not self.devices[frame.source_address].ieee_address:
            self.devices[frame.source_address].ieee_address = frame.source_ieee

        if frame.dest_address not in self.devices and frame.dest_address < 0xFFF0:
            # Don't add broadcast addresses
            self.devices[frame.dest_address] = ZigbeeDevice(
                short_address=frame.dest_address,
                ieee_address=frame.dest_ieee,
            )

    def parse_nwk_layer(self, data: bytes) -> dict[str, Any]:
        """Parse Zigbee network layer frame.

        Parses NWK frame structure including frame control, addresses,
        sequence number, radius, and optional IEEE addresses.

        Args:
            data: Raw NWK layer frame bytes.

        Returns:
            Parsed NWK layer fields including addresses and payload.

        Raises:
            ValueError: If data is too short for NWK header.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> nwk_data = bytes([0x08, 0x00, 0x00, 0x00, 0x34, 0x12, 0x1E, 0x01])
            >>> result = analyzer.parse_nwk_layer(nwk_data)
            >>> result['source_address']
            4660
        """
        if len(data) < 8:
            raise ValueError("Insufficient data for NWK header")

        # Parse frame control and basic fields
        frame_ctrl_data = self._parse_nwk_frame_control(data)

        # Parse optional fields
        offset, dest_ieee, src_ieee, security_data = self._parse_nwk_optional_fields(
            data, frame_ctrl_data, offset=8
        )

        # Extract payload
        payload = data[offset:]

        return {
            **frame_ctrl_data,
            "dest_ieee": dest_ieee,
            "source_ieee": src_ieee,
            "security_data": security_data,
            "payload": payload,
        }

    def _parse_nwk_frame_control(self, data: bytes) -> dict[str, Any]:
        """Parse NWK frame control field and basic header.

        Args:
            data: Raw NWK frame bytes.

        Returns:
            Dictionary of frame control flags and basic fields.
        """
        frame_control = int.from_bytes(data[0:2], "little")
        frame_type = (frame_control >> 0) & 0x03
        protocol_version = (frame_control >> 2) & 0x0F
        discover_route = (frame_control >> 6) & 0x03
        multicast_flag = bool((frame_control >> 8) & 0x01)
        security = bool((frame_control >> 9) & 0x01)
        source_route = bool((frame_control >> 10) & 0x01)
        dest_ieee_present = bool((frame_control >> 11) & 0x01)
        src_ieee_present = bool((frame_control >> 12) & 0x01)

        dest_addr = int.from_bytes(data[2:4], "little")
        src_addr = int.from_bytes(data[4:6], "little")
        radius = data[6]
        sequence = data[7]

        return {
            "frame_type": self.FRAME_TYPES.get(frame_type, "UNKNOWN"),
            "protocol_version": protocol_version,
            "discover_route": discover_route,
            "multicast": multicast_flag,
            "security": security,
            "source_route": source_route,
            "dest_ieee_present": dest_ieee_present,
            "src_ieee_present": src_ieee_present,
            "dest_address": dest_addr,
            "source_address": src_addr,
            "radius": radius,
            "sequence": sequence,
        }

    def _parse_nwk_optional_fields(
        self, data: bytes, frame_ctrl: dict[str, Any], offset: int
    ) -> tuple[int, int | None, int | None, dict[str, Any] | None]:
        """Parse optional NWK fields (IEEE addresses, multicast, source route, security).

        Args:
            data: Raw NWK frame bytes.
            frame_ctrl: Parsed frame control data.
            offset: Current offset in data.

        Returns:
            Tuple of (new_offset, dest_ieee, src_ieee, security_data).

        Raises:
            ValueError: If data is insufficient for optional fields.
        """
        # Optional destination IEEE address
        dest_ieee = None
        if frame_ctrl["dest_ieee_present"]:
            if len(data) < offset + 8:
                raise ValueError("Insufficient data for destination IEEE address")
            dest_ieee = int.from_bytes(data[offset : offset + 8], "little")
            offset += 8

        # Optional source IEEE address
        src_ieee = None
        if frame_ctrl["src_ieee_present"]:
            if len(data) < offset + 8:
                raise ValueError("Insufficient data for source IEEE address")
            src_ieee = int.from_bytes(data[offset : offset + 8], "little")
            offset += 8

        # Multicast control
        if frame_ctrl["multicast"]:
            if len(data) < offset + 1:
                raise ValueError("Insufficient data for multicast control")
            offset += 1

        # Source route subframe
        if frame_ctrl["source_route"]:
            if len(data) < offset + 1:
                raise ValueError("Insufficient data for source route")
            relay_count = data[offset]
            offset += 2 + (relay_count * 2)

        # Security header
        security_data = None
        if frame_ctrl["security"]:
            if len(data) < offset + 5:
                raise ValueError("Insufficient data for security header")
            security_header = parse_security_header(data[offset:])
            if "error" in security_header:
                raise ValueError(security_header["error"])
            security_data = security_header
            offset += security_header["header_length"]

        return offset, dest_ieee, src_ieee, security_data

    def parse_aps_layer(self, data: bytes) -> dict[str, Any]:
        """Parse Zigbee APS (Application Support) layer.

        Parses APS frame including frame control, addressing, and cluster ID.

        Args:
            data: APS layer frame bytes.

        Returns:
            Parsed APS layer fields.

        Raises:
            ValueError: If data is too short for APS header.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> aps_data = bytes([0x00, 0x00, 0x06, 0x00, 0x01])
            >>> result = analyzer.parse_aps_layer(aps_data)
            >>> result['cluster_id']
            6
        """
        if len(data) < 3:
            raise ValueError("Insufficient data for APS header")

        # Parse frame control
        frame_ctrl = self._parse_aps_frame_control(data[0])

        # Parse addressing and IDs
        offset, addressing_data = self._parse_aps_addressing(data, frame_ctrl, offset=1)

        # Parse extended header if present
        if frame_ctrl["extended_header"]:
            if len(data) >= offset + 1:
                offset += 1  # Skip extended frame control

        payload = data[offset:]

        cluster_id = addressing_data.get("cluster_id")
        cluster_name = (
            self.STANDARD_CLUSTERS.get(cluster_id, "Unknown")
            if cluster_id is not None and isinstance(cluster_id, int)
            else None
        )

        return {
            **frame_ctrl,
            **addressing_data,
            "cluster_name": cluster_name,
            "payload": payload,
        }

    def _parse_aps_frame_control(self, frame_control: int) -> dict[str, Any]:
        """Parse APS frame control byte.

        Args:
            frame_control: Frame control byte value.

        Returns:
            Dictionary of frame control flags.
        """
        return {
            "frame_type": frame_control & 0x03,
            "delivery_mode": (frame_control >> 2) & 0x03,
            "security": bool((frame_control >> 5) & 0x01),
            "ack_request": bool((frame_control >> 6) & 0x01),
            "extended_header": bool((frame_control >> 7) & 0x01),
        }

    def _parse_aps_addressing(
        self, data: bytes, frame_ctrl: dict[str, Any], offset: int
    ) -> tuple[int, dict[str, Any]]:
        """Parse APS addressing fields (endpoints, cluster, profile).

        Args:
            data: APS frame bytes.
            frame_ctrl: Parsed frame control data.
            offset: Current offset in data.

        Returns:
            Tuple of (new_offset, addressing_data).

        Raises:
            ValueError: If data is insufficient for addressing fields.
        """
        addressing: dict[str, Any] = {
            "dest_endpoint": None,
            "group_address": None,
            "cluster_id": None,
            "profile_id": None,
            "source_endpoint": None,
            "aps_counter": None,
        }

        delivery_mode = frame_ctrl["delivery_mode"]

        # Destination endpoint (for unicast/broadcast)
        if delivery_mode in [0x00, 0x01, 0x02]:
            if len(data) < offset + 1:
                raise ValueError("Insufficient data for destination endpoint")
            addressing["dest_endpoint"] = data[offset]
            offset += 1

        # Group address (for group delivery)
        if delivery_mode == 0x01:
            if len(data) < offset + 2:
                raise ValueError("Insufficient data for group address")
            addressing["group_address"] = int.from_bytes(data[offset : offset + 2], "little")
            offset += 2

        # Cluster ID
        if len(data) >= offset + 2:
            addressing["cluster_id"] = int.from_bytes(data[offset : offset + 2], "little")
            offset += 2

        # Profile ID
        if len(data) >= offset + 2:
            addressing["profile_id"] = int.from_bytes(data[offset : offset + 2], "little")
            offset += 2

        # Source endpoint
        if len(data) >= offset + 1:
            addressing["source_endpoint"] = data[offset]
            offset += 1

        # APS counter
        if len(data) >= offset + 1:
            addressing["aps_counter"] = data[offset]
            offset += 1

        return offset, addressing

    def parse_zcl_frame(self, cluster_id: int, data: bytes) -> dict[str, Any]:
        """Parse ZCL frame for specific cluster.

        Wrapper around zcl.parse_zcl_frame for integration with analyzer.

        Args:
            cluster_id: ZCL cluster ID.
            data: ZCL frame payload.

        Returns:
            Parsed ZCL frame data.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> zcl_data = bytes([0x01, 0x00, 0x01])
            >>> result = analyzer.parse_zcl_frame(0x0006, zcl_data)
            >>> result['command_name']
            'On'
        """
        return parse_zcl_frame(cluster_id, data)

    def discover_topology(self) -> dict[int, list[int]]:
        """Discover network topology from captured frames.

        Analyzes frame routing to determine parent-child relationships
        in the Zigbee network tree.

        Returns:
            Dictionary mapping parent addresses to list of child addresses.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> # Add frames...
            >>> topology = analyzer.discover_topology()
            >>> print(topology[0x0000])  # Coordinator's children
            [4660, 8765]
        """
        topology: dict[int, list[int]] = {}

        # Analyze frame patterns to infer topology
        # Devices that communicate frequently with coordinator are likely direct children
        # Devices that route through others are likely children of routers

        for frame in self.frames:
            # If a device sends to coordinator (0x0000), it might be a direct child
            if frame.dest_address == 0x0000 and frame.source_address not in [0x0000]:
                if 0x0000 not in topology:
                    topology[0x0000] = []
                if frame.source_address not in topology[0x0000]:
                    topology[0x0000].append(frame.source_address)

        # Infer device types from topology
        for addr, device in self.devices.items():
            if addr == 0x0000:
                device.device_type = "coordinator"
            elif addr in topology:
                device.device_type = "router"  # Has children
            else:
                device.device_type = "end_device"  # No children

            # Set parent address if known
            for parent, children in topology.items():
                if addr in children:
                    device.parent_address = parent

        return topology

    def add_network_key(self, key: bytes) -> None:
        """Add network key for decrypting secured frames.

        Args:
            key: 128-bit (16-byte) AES network key.

        Raises:
            ValueError: If key is not 16 bytes.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> key = bytes([0x01] * 16)
            >>> analyzer.add_network_key(key)
            >>> len(analyzer.network_keys)
            1
        """
        if len(key) != 16:
            raise ValueError(f"Network key must be 16 bytes, got {len(key)}")
        self.network_keys.append(key)

    def export_topology(self, output_path: Path) -> None:
        """Export network topology as JSON with device information.

        Exports topology data including devices, relationships, and
        cluster information in JSON format. Also generates GraphViz
        DOT format for visualization.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = ZigbeeAnalyzer()
            >>> # Add frames and analyze...
            >>> analyzer.export_topology(Path("zigbee_network.json"))
        """
        topology = self.discover_topology()

        export_data = {
            "devices": {
                addr: {
                    "short_address": f"0x{addr:04X}",
                    "ieee_address": f"0x{dev.ieee_address:016X}" if dev.ieee_address else None,
                    "device_type": dev.device_type,
                    "parent_address": f"0x{dev.parent_address:04X}" if dev.parent_address else None,
                    "clusters": [
                        {
                            "id": f"0x{cid:04X}",
                            "name": self.STANDARD_CLUSTERS.get(cid, "Unknown"),
                        }
                        for cid in dev.clusters
                    ],
                    "manufacturer": dev.manufacturer,
                    "model": dev.model,
                }
                for addr, dev in self.devices.items()
            },
            "topology": {
                f"0x{parent:04X}": [f"0x{child:04X}" for child in children]
                for parent, children in topology.items()
            },
        }

        # Write JSON
        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        # Generate GraphViz DOT file
        dot_path = output_path.with_suffix(".dot")
        with dot_path.open("w") as f:
            f.write("digraph ZigbeeNetwork {\n")
            f.write("  rankdir=TB;\n")
            f.write("  node [shape=box];\n\n")

            # Add nodes
            for addr, dev in self.devices.items():
                label = f"0x{addr:04X}\\n{dev.device_type}"
                if dev.manufacturer:
                    label += f"\\n{dev.manufacturer}"
                if dev.model:
                    label += f"\\n{dev.model}"

                shape = "ellipse" if dev.device_type == "coordinator" else "box"
                f.write(f'  "0x{addr:04X}" [label="{label}", shape={shape}];\n')

            f.write("\n")

            # Add edges
            for parent, children in topology.items():
                for child in children:
                    f.write(f'  "0x{parent:04X}" -> "0x{child:04X}";\n')

            f.write("}\n")


__all__ = ["ZigbeeAnalyzer", "ZigbeeDevice", "ZigbeeFrame"]
