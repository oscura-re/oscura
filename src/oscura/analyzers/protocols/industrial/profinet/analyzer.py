"""PROFINET IO protocol analyzer.

This module provides comprehensive PROFINET protocol analysis for real-time
industrial Ethernet communication. Supports RT (Real-Time), IRT (Isochronous
Real-Time), DCP (Discovery and Configuration), and PTCP (Precision Time) protocols.

Example:
    >>> from oscura.analyzers.protocols.industrial.profinet import ProfinetAnalyzer
    >>> analyzer = ProfinetAnalyzer()
    >>> # Parse Ethernet frame containing PROFINET data
    >>> frame = analyzer.parse_frame(ethernet_frame, timestamp=0.0)
    >>> print(f"Frame Type: {frame.frame_type}, Frame ID: 0x{frame.frame_id:04X}")
    >>> # Discover devices from DCP Identify responses
    >>> devices = analyzer.discover_devices()
    >>> for device in devices:
    ...     print(f"Device: {device.device_name} at {device.mac_address}")
    >>> # Export topology
    >>> analyzer.export_topology(Path("profinet_topology.json"))

References:
    PROFINET Specification V2.4:
    https://www.profibus.com/download/profinet-specification/

    IEC 61158 / IEC 61784 (Industrial communication networks)
    Wireshark PROFINET dissector
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.analyzers.protocols.industrial.profinet.dcp import DCPParser
from oscura.analyzers.protocols.industrial.profinet.ptcp import PTCPParser


@dataclass
class ProfinetFrame:
    """PROFINET frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        frame_type: Frame type name (e.g., "RT_CLASS_1", "DCP", "PTCP").
        frame_id: PROFINET frame ID (0x8000-0xFFFF).
        source_mac: Source MAC address (XX:XX:XX:XX:XX:XX format).
        dest_mac: Destination MAC address.
        cycle_counter: Cycle counter for RT frames.
        data_status: Data status byte for RT frames.
        payload: Raw payload bytes.
        decoded: Decoded frame data (varies by frame type).
    """

    timestamp: float
    frame_type: str
    frame_id: int
    source_mac: str
    dest_mac: str
    cycle_counter: int | None = None
    data_status: int | None = None
    payload: bytes = b""
    decoded: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfinetDevice:
    """PROFINET device information.

    Attributes:
        mac_address: Device MAC address.
        device_name: Device name from DCP (Name of Station).
        device_type: Device type description.
        vendor_id: Vendor ID from DCP Device ID block.
        device_id: Device ID from DCP Device ID block.
        station_type: Station type (IO-Controller, IO-Device, etc.).
        modules: List of device modules/submodules.
        ip_address: Device IP address (if configured).
        subnet_mask: Subnet mask.
        gateway: Gateway address.
    """

    mac_address: str
    device_name: str | None = None
    device_type: str | None = None
    vendor_id: int | None = None
    device_id: int | None = None
    station_type: str = "DEVICE"
    modules: list[dict[str, Any]] = field(default_factory=list)
    ip_address: str | None = None
    subnet_mask: str | None = None
    gateway: str | None = None


class ProfinetAnalyzer:
    """PROFINET IO protocol analyzer.

    Provides comprehensive analysis of PROFINET frames including RT (Real-Time),
    IRT (Isochronous Real-Time), DCP (Discovery and Configuration Protocol),
    and PTCP (Precision Transparent Clock Protocol).

    Attributes:
        frames: List of parsed PROFINET frames.
        devices: Dictionary of discovered devices by MAC address.

    Example:
        >>> analyzer = ProfinetAnalyzer()
        >>> # Parse frame from raw Ethernet data
        >>> frame = analyzer.parse_frame(eth_frame, timestamp=1.234)
        >>> # Discover devices
        >>> devices = analyzer.discover_devices()
        >>> print(f"Found {len(devices)} PROFINET devices")
    """

    # PROFINET Frame ID ranges (0x0000-0xFFFF)
    FRAME_ID_RANGES: ClassVar[list[tuple[tuple[int, int], str]]] = [
        ((0x0000, 0x7FFF), "Reserved"),
        ((0x8000, 0xBFFF), "RT_CLASS_1"),  # Cyclic Real-Time data
        ((0xC000, 0xFBFF), "RT_CLASS_UDP"),  # RT over UDP
        ((0xFC00, 0xFCFF), "RT_CLASS_2"),  # IRT
        ((0xFD00, 0xFDFF), "RT_CLASS_3"),  # IRT with fragmentation
        ((0xFE00, 0xFEFF), "Reserved for profiles"),
        ((0xFF00, 0xFF1F), "Multicast MAC range"),
        ((0xFF20, 0xFF3F), "PTCP"),  # Precision Time Protocol - Delay
        ((0xFF40, 0xFF4F), "PTCP"),  # Precision Time Protocol - Sync
        ((0xFF50, 0xFF8F), "PTCP"),  # Reserved for PTCP
        ((0xFF90, 0xFFFF), "Reserved"),
    ]

    # PROFINET EtherType
    ETHERTYPE_PROFINET: ClassVar[int] = 0x8892

    def __init__(self) -> None:
        """Initialize PROFINET analyzer."""
        self.frames: list[ProfinetFrame] = []
        self.devices: dict[str, ProfinetDevice] = {}
        self._dcp_parser = DCPParser()
        self._ptcp_parser = PTCPParser()

    def parse_frame(self, ethernet_frame: bytes, timestamp: float = 0.0) -> ProfinetFrame:
        """Parse PROFINET frame from Ethernet payload.

        Extracts PROFINET-specific data from raw Ethernet frame and decodes
        based on frame type (RT, DCP, PTCP, etc.).

        Ethernet Frame Format:
        - Destination MAC (6 bytes)
        - Source MAC (6 bytes)
        - EtherType/Length (2 bytes) - should be 0x8892 for PROFINET
        - PROFINET Data (variable)
        - FCS (4 bytes, typically stripped by capture)

        Args:
            ethernet_frame: Complete Ethernet frame including headers.
            timestamp: Frame timestamp in seconds.

        Returns:
            Parsed PROFINET frame.

        Raises:
            ValueError: If frame is too short or not a valid PROFINET frame.

        Example:
            >>> analyzer = ProfinetAnalyzer()
            >>> eth_frame = bytes([...])  # Raw Ethernet frame
            >>> frame = analyzer.parse_frame(eth_frame, timestamp=1.5)
            >>> print(f"Frame type: {frame.frame_type}")
        """
        if len(ethernet_frame) < 14:
            raise ValueError(f"Ethernet frame too short: {len(ethernet_frame)} bytes (minimum 14)")

        # Parse Ethernet header and extract PROFINET payload
        dest_mac, source_mac, profinet_data = self._parse_ethernet_header(ethernet_frame)

        # Extract Frame ID and classify
        frame_id = int.from_bytes(profinet_data[0:2], "big")
        frame_type = self._classify_frame_id(frame_id)
        frame_payload = profinet_data[2:]

        # Decode frame based on type (may update frame_type for DCP)
        frame_type, decoded, cycle_counter, data_status = self._decode_frame_payload(
            frame_id, frame_type, frame_payload
        )

        frame = ProfinetFrame(
            timestamp=timestamp,
            frame_type=frame_type,
            frame_id=frame_id,
            source_mac=source_mac,
            dest_mac=dest_mac,
            cycle_counter=cycle_counter,
            data_status=data_status,
            payload=frame_payload,
            decoded=decoded,
        )

        self.frames.append(frame)
        self._update_device_info(frame)
        return frame

    def _parse_ethernet_header(self, ethernet_frame: bytes) -> tuple[str, str, bytes]:
        """Parse Ethernet header and extract PROFINET payload."""
        dest_mac = ":".join(f"{b:02x}" for b in ethernet_frame[0:6])
        source_mac = ":".join(f"{b:02x}" for b in ethernet_frame[6:12])
        ethertype = int.from_bytes(ethernet_frame[12:14], "big")

        # Handle VLAN-tagged frames
        payload_offset = 14
        if ethertype == 0x8100:
            if len(ethernet_frame) < 18:
                raise ValueError("VLAN-tagged frame too short")
            ethertype = int.from_bytes(ethernet_frame[16:18], "big")
            payload_offset = 18

        if ethertype != self.ETHERTYPE_PROFINET:
            raise ValueError(
                f"Not a PROFINET frame: EtherType 0x{ethertype:04X} "
                f"(expected 0x{self.ETHERTYPE_PROFINET:04X})"
            )

        profinet_data = ethernet_frame[payload_offset:]
        if len(profinet_data) < 2:
            raise ValueError(f"PROFINET data too short: {len(profinet_data)} bytes")

        return dest_mac, source_mac, profinet_data

    def _decode_frame_payload(
        self, frame_id: int, frame_type: str, frame_payload: bytes
    ) -> tuple[str, dict[str, Any], int | None, int | None]:
        """Decode frame payload based on frame type.

        Returns:
            Tuple of (frame_type, decoded, cycle_counter, data_status).
            frame_type may be updated for DCP frames.
        """
        decoded: dict[str, Any] = {}
        cycle_counter, data_status = None, None

        if frame_type.startswith("RT_CLASS"):
            decoded = self._parse_rt_frame(frame_id, frame_payload)
            cycle_counter = decoded.get("cycle_counter")
            data_status = decoded.get("data_status")

        elif frame_type == "PTCP":
            try:
                decoded = self._ptcp_parser.parse_frame(frame_id, frame_payload)
            except ValueError as e:
                decoded = {"parse_error": str(e)}

        elif frame_id in (0xFEFC, 0xFEFD):
            frame_type = "DCP"
            try:
                decoded = self._dcp_parser.parse_frame(frame_payload)
            except ValueError as e:
                decoded = {"parse_error": str(e)}

        return frame_type, decoded, cycle_counter, data_status

    def _classify_frame_id(self, frame_id: int) -> str:
        """Classify frame based on Frame ID range.

        Args:
            frame_id: PROFINET frame ID (0x0000-0xFFFF).

        Returns:
            Frame type classification string.
        """
        for (start, end), frame_type in self.FRAME_ID_RANGES:
            if start <= frame_id <= end:
                return frame_type
        return f"Unknown (0x{frame_id:04X})"

    def _parse_rt_frame(self, frame_id: int, data: bytes) -> dict[str, Any]:
        """Parse PROFINET Real-Time frame.

        RT Frame Format:
        - Cycle Counter (2 bytes) - for RT_CLASS_1 and higher
        - Data Status (1 byte)
        - Transfer Status (1 byte) - optional for some classes
        - I/O Data (variable)

        Args:
            frame_id: PROFINET frame ID.
            data: RT frame payload.

        Returns:
            Parsed RT frame data.
        """
        if len(data) < 4:
            return {"error": "RT frame too short", "raw_data": data.hex()}

        result: dict[str, Any] = {"frame_id": frame_id}

        # Check if this is cyclic data (has cycle counter)
        if 0x8000 <= frame_id <= 0xFDFF:
            cycle_counter = int.from_bytes(data[0:2], "big")
            data_status = data[2]
            io_data = data[3:]

            # Parse Data Status byte (IEC 61158-6-10)
            # Bit 7: State (0=BACKUP, 1=PRIMARY)
            # Bit 6: Redundancy (0=No redundancy, 1=Redundancy enabled)
            # Bit 5: DataValid (0=Invalid, 1=Valid)
            # Bit 4: Reserved
            # Bit 3: Provider State (0=STOP, 1=RUN)
            # Bit 2: Station Problem Indicator (0=Normal, 1=Problem)
            # Bit 1: Reserved
            # Bit 0: Reserved

            result.update(
                {
                    "cycle_counter": cycle_counter,
                    "data_status": data_status,
                    "data_status_flags": {
                        "primary": bool(data_status & 0x80),
                        "redundancy": bool(data_status & 0x40),
                        "data_valid": bool(data_status & 0x20),
                        "provider_state": "RUN" if (data_status & 0x08) else "STOP",
                        "station_problem": bool(data_status & 0x04),
                    },
                    "io_data": io_data.hex(),
                    "io_data_length": len(io_data),
                }
            )

        return result

    def _update_device_info(self, frame: ProfinetFrame) -> None:
        """Update device information from parsed frame.

        Extracts and updates device information from DCP frames.

        Args:
            frame: Parsed PROFINET frame.
        """
        if frame.frame_type != "DCP" or "blocks" not in frame.decoded:
            return

        # Extract device information from DCP blocks
        mac = frame.source_mac
        if mac not in self.devices:
            self.devices[mac] = ProfinetDevice(mac_address=mac)

        device = self.devices[mac]

        for block in frame.decoded.get("blocks", []):
            # Device name
            if "device_name" in block:
                device.device_name = block["device_name"]

            # Device ID (Vendor ID + Device ID)
            if "vendor_id" in block and "device_id" in block:
                device.vendor_id = block["vendor_id"]
                device.device_id = block["device_id"]

            # Device role
            if "role_names" in block:
                roles = block["role_names"]
                if "IO-Controller" in roles:
                    device.station_type = "IO-Controller"
                elif "IO-Device" in roles:
                    device.station_type = "IO-Device"
                elif "IO-Supervisor" in roles:
                    device.station_type = "IO-Supervisor"

            # IP configuration
            if "ip_address" in block:
                device.ip_address = block["ip_address"]
            if "subnet_mask" in block:
                device.subnet_mask = block["subnet_mask"]
            if "gateway" in block:
                device.gateway = block["gateway"]

    def discover_devices(self) -> list[ProfinetDevice]:
        """Discover PROFINET devices from DCP Identify responses.

        Analyzes all parsed frames to extract device information from
        DCP protocol messages.

        Returns:
            List of discovered PROFINET devices.

        Example:
            >>> analyzer = ProfinetAnalyzer()
            >>> # ... parse frames ...
            >>> devices = analyzer.discover_devices()
            >>> for device in devices:
            ...     print(f"{device.mac_address}: {device.device_name}")
        """
        return list(self.devices.values())

    def export_topology(self, output_path: Path) -> None:
        """Export device topology as JSON.

        Exports all discovered devices and their configuration to a JSON file.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = ProfinetAnalyzer()
            >>> # ... parse frames and discover devices ...
            >>> analyzer.export_topology(Path("profinet_network.json"))
        """
        topology = {
            "network_type": "PROFINET IO",
            "devices": [
                {
                    "mac_address": device.mac_address,
                    "device_name": device.device_name,
                    "device_type": device.device_type,
                    "vendor_id": device.vendor_id,
                    "device_id": device.device_id,
                    "station_type": device.station_type,
                    "ip_address": device.ip_address,
                    "subnet_mask": device.subnet_mask,
                    "gateway": device.gateway,
                    "modules": device.modules,
                }
                for device in self.devices.values()
            ],
            "frame_count": len(self.frames),
            "frame_types": self._get_frame_type_statistics(),
        }

        with output_path.open("w") as f:
            json.dump(topology, f, indent=2)

    def _get_frame_type_statistics(self) -> dict[str, int]:
        """Get statistics of frame types seen.

        Returns:
            Dictionary mapping frame type to count.
        """
        stats: dict[str, int] = {}
        for frame in self.frames:
            stats[frame.frame_type] = stats.get(frame.frame_type, 0) + 1
        return stats


__all__ = ["ProfinetAnalyzer", "ProfinetDevice", "ProfinetFrame"]
