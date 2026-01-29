"""EtherCAT protocol analyzer.

This module provides comprehensive EtherCAT (Ethernet for Control Automation Technology)
protocol analysis supporting frame parsing, datagram decoding, topology discovery,
and slave configuration export.

Example:
    >>> from oscura.analyzers.protocols.industrial.ethercat.analyzer import EtherCATAnalyzer
    >>> analyzer = EtherCATAnalyzer()
    >>> # Parse EtherCAT frame (Ethertype 0x88A4)
    >>> ethernet_payload = bytes([0x11, 0x10, 0x01, 0x00, ...])
    >>> frame = analyzer.parse_frame(ethernet_payload, timestamp=0.0)
    >>> print(f"Datagrams: {len(frame.datagrams)}")
    >>> # Access slave information
    >>> slave = analyzer.read_slave_info(station_address=1)
    >>> if slave:
    ...     print(f"State: {slave.state}")

References:
    IEC 61158 Type 12: Industrial communication networks
    ETG.1000 EtherCAT Protocol Specification
    ETG.2000 EtherCAT AL Protocol (Application Layer)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class EtherCATDatagram:
    """EtherCAT datagram within frame.

    A single EtherCAT frame can contain multiple datagrams. Each datagram
    represents a command (read/write) to EtherCAT slave devices.

    Attributes:
        cmd: Command type code (0x00-0x0E).
        cmd_name: Human-readable command name.
        idx: Index field (lower 4 bits of first byte).
        adp: Auto-increment address (2 bytes).
        ado: Address offset (2 bytes).
        len_: Data length (11 bits).
        irq: Interrupt requested (4 bits).
        data: Datagram payload data.
        wkc: Working counter (2 bytes).
        more_follows: True if more datagrams follow in frame.
    """

    cmd: int
    cmd_name: str
    idx: int
    adp: int
    ado: int
    len_: int
    irq: int
    data: bytes
    wkc: int
    more_follows: bool


@dataclass
class EtherCATFrame:
    """EtherCAT frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        length: Frame length field (11 bits).
        datagrams: List of datagrams in frame.
    """

    timestamp: float
    length: int
    datagrams: list[EtherCATDatagram]


@dataclass
class EtherCATSlave:
    """EtherCAT slave device.

    Attributes:
        station_address: Configured station address.
        alias_address: Alias address (if configured).
        vendor_id: Vendor identification.
        product_code: Product code.
        revision: Revision number.
        serial_number: Serial number.
        state: Current slave state (INIT, PRE-OP, SAFE-OP, OP).
        dc_supported: Distributed Clock support.
        mailbox_protocols: Supported mailbox protocols (CoE, FoE, SoE, EoE).
    """

    station_address: int
    alias_address: int | None = None
    vendor_id: int | None = None
    product_code: int | None = None
    revision: int | None = None
    serial_number: int | None = None
    state: str = "UNKNOWN"
    dc_supported: bool = False
    mailbox_protocols: list[str] = field(default_factory=list)


class EtherCATAnalyzer:
    """EtherCAT protocol analyzer.

    Provides comprehensive EtherCAT protocol analysis including frame parsing,
    datagram decoding, topology discovery, and slave state tracking.

    Attributes:
        frames: List of parsed EtherCAT frames.
        slaves: Dictionary of discovered slaves by station address.

    Example:
        >>> analyzer = EtherCATAnalyzer()
        >>> # Parse frame
        >>> frame = analyzer.parse_frame(ethernet_payload, timestamp=0.0)
        >>> # Discover topology
        >>> slave_addresses = analyzer.discover_topology()
        >>> # Export configuration
        >>> from pathlib import Path
        >>> analyzer.export_configuration(Path("ethercat_config.xml"))
    """

    # Command types (EtherCAT datagram commands)
    COMMANDS: ClassVar[dict[int, str]] = {
        0x00: "NOP",
        0x01: "APRD",  # Auto-increment Physical Read
        0x02: "APWR",  # Auto-increment Physical Write
        0x03: "APRW",  # Auto-increment Physical Read/Write
        0x04: "FPRD",  # Configured address Physical Read
        0x05: "FPWR",  # Configured address Physical Write
        0x06: "FPRW",  # Configured address Physical Read/Write
        0x07: "BRD",  # Broadcast Read
        0x08: "BWR",  # Broadcast Write
        0x09: "BRW",  # Broadcast Read/Write
        0x0A: "LRD",  # Logical Memory Read
        0x0B: "LWR",  # Logical Memory Write
        0x0C: "LRW",  # Logical Memory Read/Write
        0x0D: "ARMW",  # Auto-increment physical Read Multiple Write
        0x0E: "FRMW",  # Configured address physical Read Multiple Write
    }

    # Slave state machine states
    STATES: ClassVar[dict[int, str]] = {
        0x01: "INIT",
        0x02: "PRE-OP",
        0x04: "SAFE-OP",
        0x08: "OP",
    }

    # Mailbox protocol types
    MAILBOX_PROTOCOLS: ClassVar[dict[int, str]] = {
        0x02: "CoE",  # CAN application protocol over EtherCAT
        0x03: "FoE",  # File access over EtherCAT
        0x04: "SoE",  # Servo drive profile over EtherCAT
        0x05: "EoE",  # Ethernet over EtherCAT
    }

    # Well-known EtherCAT register addresses
    REG_TYPE: ClassVar[int] = 0x0000  # Type
    REG_REVISION: ClassVar[int] = 0x0001  # Revision
    REG_BUILD: ClassVar[int] = 0x0002  # Build
    REG_FMMU_COUNT: ClassVar[int] = 0x0004  # FMMU count
    REG_SYNC_COUNT: ClassVar[int] = 0x0005  # SyncManager count
    REG_PORT_DESC: ClassVar[int] = 0x0007  # Port descriptor
    REG_ESC_FEATURES: ClassVar[int] = 0x0008  # ESC features
    REG_STATION_ADDRESS: ClassVar[int] = 0x0010  # Configured Station Address
    REG_ALIAS_ADDRESS: ClassVar[int] = 0x0012  # Alias Address
    REG_DL_CONTROL: ClassVar[int] = 0x0100  # DL Control
    REG_DL_STATUS: ClassVar[int] = 0x0110  # DL Status
    REG_AL_CONTROL: ClassVar[int] = 0x0120  # AL Control
    REG_AL_STATUS: ClassVar[int] = 0x0130  # AL Status
    REG_AL_STATUS_CODE: ClassVar[int] = 0x0134  # AL Status Code
    REG_DC_SYSTEM_TIME: ClassVar[int] = 0x0910  # Distributed Clock System Time

    def __init__(self) -> None:
        """Initialize EtherCAT analyzer."""
        self.frames: list[EtherCATFrame] = []
        self.slaves: dict[int, EtherCATSlave] = {}

    def parse_frame(self, ethernet_payload: bytes, timestamp: float = 0.0) -> EtherCATFrame:
        """Parse EtherCAT frame (Ethertype 0x88A4).

        Frame Format:
        - Length (2 bytes) - lower 11 bits, upper 5 bits reserved
        - Type (1 byte) - Protocol type (0x01 for EtherCAT commands)
        - Datagrams (variable) - One or more datagrams

        Args:
            ethernet_payload: EtherCAT frame payload (after Ethernet header).
            timestamp: Frame timestamp in seconds.

        Returns:
            Parsed EtherCAT frame.

        Raises:
            ValueError: If frame is invalid.

        Example:
            >>> analyzer = EtherCATAnalyzer()
            >>> # Simple frame with one datagram
            >>> payload = bytes([0x0E, 0x10, 0x01, 0x00, 0x00, 0x00, ...])
            >>> frame = analyzer.parse_frame(payload, timestamp=0.0)
            >>> assert len(frame.datagrams) >= 1
        """
        if len(ethernet_payload) < 2:
            raise ValueError(f"EtherCAT frame too short: {len(ethernet_payload)} bytes (minimum 2)")

        # Parse length field (11 bits)
        length_and_reserved = int.from_bytes(ethernet_payload[0:2], "little")
        length = length_and_reserved & 0x07FF

        datagrams: list[EtherCATDatagram] = []
        offset = 2

        # Parse all datagrams
        while offset < len(ethernet_payload):
            try:
                datagram, consumed = self._parse_datagram(ethernet_payload, offset)
                datagrams.append(datagram)
                offset += consumed

                if not datagram.more_follows:
                    break
            except ValueError:
                # If we fail to parse a datagram but have already parsed some, stop gracefully
                if len(datagrams) > 0:
                    break
                raise

        frame = EtherCATFrame(
            timestamp=timestamp,
            length=length,
            datagrams=datagrams,
        )

        self.frames.append(frame)
        self._update_slave_state(frame)

        return frame

    def _parse_datagram(self, data: bytes, offset: int) -> tuple[EtherCATDatagram, int]:
        """Parse single EtherCAT datagram.

        Datagram Format:
        - Cmd (1 byte) - Command type
        - Idx (1 byte) - Index
        - ADP (2 bytes) - Auto-increment address (little-endian)
        - ADO (2 bytes) - Address offset (little-endian)
        - Len/M/IRQ (2 bytes) - Length (11 bits), More (1 bit), IRQ (4 bits)
        - Data (Len bytes)
        - WKC (2 bytes) - Working counter (little-endian)

        Args:
            data: Frame data containing datagram.
            offset: Offset to start of datagram.

        Returns:
            Tuple of (datagram, bytes_consumed).

        Raises:
            ValueError: If datagram is invalid.
        """
        if offset + 10 > len(data):
            raise ValueError(
                f"Insufficient data for datagram header at offset {offset}: "
                f"need 10 bytes, have {len(data) - offset}"
            )

        cmd = data[offset]
        idx = data[offset + 1]
        adp = int.from_bytes(data[offset + 2 : offset + 4], "little")
        ado = int.from_bytes(data[offset + 4 : offset + 6], "little")
        len_m_irq = int.from_bytes(data[offset + 6 : offset + 8], "little")

        # Extract fields from len_m_irq
        data_len = len_m_irq & 0x07FF  # Lower 11 bits
        more_follows = bool(len_m_irq & 0x8000)  # Bit 15
        irq = (len_m_irq >> 12) & 0x0F  # Bits 12-15 (but bit 15 is M flag)

        # Validate data length
        if offset + 10 + data_len > len(data):
            raise ValueError(
                f"Insufficient data for datagram payload at offset {offset}: "
                f"need {data_len} bytes, have {len(data) - offset - 10}"
            )

        datagram_data = data[offset + 8 : offset + 8 + data_len]
        wkc = int.from_bytes(data[offset + 8 + data_len : offset + 10 + data_len], "little")

        total_consumed = 10 + data_len

        datagram = EtherCATDatagram(
            cmd=cmd,
            cmd_name=self.COMMANDS.get(cmd, f"Unknown (0x{cmd:02X})"),
            idx=idx,
            adp=adp,
            ado=ado,
            len_=data_len,
            irq=irq,
            data=datagram_data,
            wkc=wkc,
            more_follows=more_follows,
        )

        return datagram, total_consumed

    def _update_slave_state(self, frame: EtherCATFrame) -> None:
        """Update slave state based on parsed frame.

        Args:
            frame: Parsed EtherCAT frame.
        """
        for datagram in frame.datagrams:
            # Check for AL Status register reads (address 0x0130)
            if datagram.ado == self.REG_AL_STATUS and len(datagram.data) >= 2:
                state_code = datagram.data[0] & 0x0F  # Lower 4 bits
                state_name = self.STATES.get(state_code, "UNKNOWN")

                # For auto-increment addressing, adp indicates position
                if datagram.cmd in {0x01, 0x02, 0x03}:  # APRD, APWR, APRW
                    station_address = datagram.adp
                    if station_address not in self.slaves:
                        self.slaves[station_address] = EtherCATSlave(
                            station_address=station_address
                        )
                    self.slaves[station_address].state = state_name

            # Check for configured station address reads (address 0x0010)
            if datagram.ado == self.REG_STATION_ADDRESS and len(datagram.data) >= 2:
                station_address = int.from_bytes(datagram.data[0:2], "little")
                if station_address not in self.slaves and station_address != 0:
                    self.slaves[station_address] = EtherCATSlave(station_address=station_address)

    def discover_topology(self) -> list[int]:
        """Discover slave topology using auto-increment addressing.

        Uses the auto-increment addressing mechanism to enumerate all slaves
        in the EtherCAT segment. This method analyzes already-parsed frames
        to extract slave addresses.

        Returns:
            List of discovered slave station addresses.

        Example:
            >>> analyzer = EtherCATAnalyzer()
            >>> # Parse some frames first...
            >>> addresses = analyzer.discover_topology()
            >>> print(f"Found {len(addresses)} slaves")
        """
        discovered: set[int] = set()

        for frame in self.frames:
            for datagram in frame.datagrams:
                # Auto-increment commands (APRD, APWR, APRW)
                if datagram.cmd in {0x01, 0x02, 0x03}:
                    # ADP field is the position, WKC indicates success
                    if datagram.wkc > 0:
                        discovered.add(datagram.adp)

                # Configured address reads
                elif datagram.cmd in {0x04, 0x05, 0x06}:  # FPRD, FPWR, FPRW
                    if datagram.wkc > 0:
                        discovered.add(datagram.adp)

        return sorted(discovered)

    def read_slave_info(self, station_address: int) -> EtherCATSlave | None:
        """Read slave information from analysis.

        Args:
            station_address: Slave station address.

        Returns:
            Slave information if found, None otherwise.

        Example:
            >>> analyzer = EtherCATAnalyzer()
            >>> # After parsing frames...
            >>> slave = analyzer.read_slave_info(station_address=1)
            >>> if slave:
            ...     print(f"State: {slave.state}")
        """
        return self.slaves.get(station_address)

    def export_configuration(self, output_path: Path) -> None:
        """Export slave configuration as ENI (EtherCAT Network Information) XML.

        Creates a simplified ENI XML file containing discovered slave configuration.
        This is compatible with EtherCAT master implementations.

        Args:
            output_path: Path to output XML file.

        Example:
            >>> analyzer = EtherCATAnalyzer()
            >>> # After parsing frames...
            >>> from pathlib import Path
            >>> analyzer.export_configuration(Path("ethercat_network.xml"))
        """
        # Create root element
        root = ET.Element("EtherCATConfig")
        root.set("Version", "1.0")

        # Create Config section
        config = ET.SubElement(root, "Config")

        # Create Master section
        master = ET.SubElement(config, "Master")

        # Add slaves
        for slave in sorted(self.slaves.values(), key=lambda s: s.station_address):
            slave_elem = ET.SubElement(master, "Slave")

            # Add slave information
            info = ET.SubElement(slave_elem, "Info")

            station = ET.SubElement(info, "StationAddress")
            station.text = str(slave.station_address)

            if slave.alias_address is not None:
                alias = ET.SubElement(info, "AliasAddress")
                alias.text = str(slave.alias_address)

            if slave.vendor_id is not None:
                vendor = ET.SubElement(info, "VendorId")
                vendor.text = f"0x{slave.vendor_id:08X}"

            if slave.product_code is not None:
                product = ET.SubElement(info, "ProductCode")
                product.text = f"0x{slave.product_code:08X}"

            if slave.revision is not None:
                revision = ET.SubElement(info, "Revision")
                revision.text = f"0x{slave.revision:08X}"

            if slave.serial_number is not None:
                serial = ET.SubElement(info, "SerialNumber")
                serial.text = str(slave.serial_number)

            # Add state information
            state = ET.SubElement(info, "State")
            state.text = slave.state

            # Add DC support
            if slave.dc_supported:
                dc = ET.SubElement(info, "Dc")
                dc.set("supported", "true")

            # Add mailbox protocols
            if slave.mailbox_protocols:
                mailbox = ET.SubElement(info, "Mailbox")
                for protocol in slave.mailbox_protocols:
                    proto_elem = ET.SubElement(mailbox, "Protocol")
                    proto_elem.text = protocol

        # Create ElementTree and write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")

        with output_path.open("wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)


__all__ = [
    "EtherCATAnalyzer",
    "EtherCATDatagram",
    "EtherCATFrame",
    "EtherCATSlave",
]
